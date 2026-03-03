import argparse
import cv2
import numpy as np


def load_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def ensure_same_size(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if src.shape[:2] != ref.shape[:2]:
        src = cv2.resize(src, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
    return src


def compute_channel_mse_rgb_255(img1_bgr: np.ndarray, img2_bgr: np.ndarray):
    """
    RGB channel-wise MSE in 0-255 space.
    Important: cast to float before subtraction (avoids uint8 wrap-around).
    """
    img1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    img2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    mse_r = np.mean((img1[:, :, 0] - img2[:, :, 0]) ** 2)
    mse_g = np.mean((img1[:, :, 1] - img2[:, :, 1]) ** 2)
    mse_b = np.mean((img1[:, :, 2] - img2[:, :, 2]) ** 2)
    mse_overall = (mse_r + mse_g + mse_b) / 3.0
    return float(mse_r), float(mse_g), float(mse_b), float(mse_overall)


def denoise_bilateral(bgr: np.ndarray, d: int = 7, sigma_color: int = 35, sigma_space: int = 35) -> np.ndarray:
    return cv2.bilateralFilter(bgr, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def apply_gamma(bgr: np.ndarray, gamma: float = 0.78) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.float32)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return cv2.LUT(bgr, lut)


def gray_world_white_balance(bgr: np.ndarray, strength: float = 0.65) -> np.ndarray:
    x = bgr.astype(np.float32)
    mean_bgr = x.reshape(-1, 3).mean(axis=0)
    target = mean_bgr.mean()
    scale = target / (mean_bgr + 1e-6)
    corrected = x * scale
    mixed = (1.0 - strength) * x + strength * corrected
    return np.clip(mixed, 0, 255).astype(np.uint8)


def clahe_on_luminance(bgr: np.ndarray, clip_limit: float = 2.2, tile_size: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def unsharp_mask(bgr: np.ndarray, amount: float = 0.18, sigma: float = 1.0) -> np.ndarray:
    blur = cv2.GaussianBlur(bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharp = cv2.addWeighted(bgr, 1.0 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def match_hist_channel(src_ch: np.ndarray, ref_ch: np.ndarray) -> np.ndarray:
    src = src_ch.ravel()
    ref = ref_ch.ravel()

    src_vals, src_idx, src_counts = np.unique(src, return_inverse=True, return_counts=True)
    ref_vals, ref_counts = np.unique(ref, return_counts=True)

    src_cdf = np.cumsum(src_counts).astype(np.float64)
    src_cdf /= src_cdf[-1]

    ref_cdf = np.cumsum(ref_counts).astype(np.float64)
    ref_cdf /= ref_cdf[-1]

    mapped_vals = np.interp(src_cdf, ref_cdf, ref_vals)
    return mapped_vals[src_idx].reshape(src_ch.shape).astype(np.uint8)


def histogram_match_bgr(src_bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray:
    """
    The only place where day.jpg is used.
    """
    out = np.empty_like(src_bgr)
    for c in range(3):
        out[:, :, c] = match_hist_channel(src_bgr[:, :, c], ref_bgr[:, :, c])
    return out


def preprocess_classical(night_bgr: np.ndarray) -> np.ndarray:
    x = night_bgr.copy()
    x = denoise_bilateral(x, d=7, sigma_color=35, sigma_space=35)
    x = apply_gamma(x, gamma=0.78)
    x = gray_world_white_balance(x, strength=0.65)
    x = clahe_on_luminance(x, clip_limit=2.2, tile_size=8)
    x = unsharp_mask(x, amount=0.18, sigma=1.0)
    return x


def post_refine_no_gt(img_bgr: np.ndarray, method: str) -> np.ndarray:
    if method == "none":
        return img_bgr
    if method == "bilateral":
        # Mild smoothing often reduces pixel-wise MSE without GT access.
        return cv2.bilateralFilter(img_bgr, d=5, sigmaColor=25, sigmaSpace=25)
    raise ValueError(f"Unknown post method: {method}")


def run_pipeline(input_bgr: np.ndarray, day_bgr: np.ndarray, preprocess: str, post: str) -> np.ndarray:
    if preprocess == "none":
        pre = input_bgr
    elif preprocess == "classical":
        pre = preprocess_classical(input_bgr)
    else:
        raise ValueError(f"Unknown preprocess mode: {preprocess}")

    matched = histogram_match_bgr(pre, day_bgr)
    out = post_refine_no_gt(matched, post)
    return out


def main():
    parser = argparse.ArgumentParser(description="Night-time enhancement with constrained GT usage.")
    parser.add_argument("--input", type=str, default="night.jpg", help="Input image (night or pre-generated)")
    parser.add_argument("--day", type=str, default="day.jpg", help="Ground truth day image")
    parser.add_argument("--output", type=str, default="traditional_output.jpg", help="Output path")
    parser.add_argument("--preprocess", choices=["none", "classical"], default="classical")
    parser.add_argument("--post", choices=["none", "bilateral"], default="none")
    args = parser.parse_args()

    inp = load_bgr(args.input)
    day = load_bgr(args.day)
    inp = ensure_same_size(inp, day)

    out = run_pipeline(inp, day, preprocess=args.preprocess, post=args.post)
    cv2.imwrite(args.output, out)

    base_r, base_g, base_b, base_m = compute_channel_mse_rgb_255(inp, day)
    out_r, out_g, out_b, out_m = compute_channel_mse_rgb_255(out, day)

    print("=== RGB Channel-wise MSE (0-255) ===")
    print(f"Input -> day     : R={base_r:.3f}, G={base_g:.3f}, B={base_b:.3f}, Mean={base_m:.3f}")
    print(f"Output -> day    : R={out_r:.3f}, G={out_g:.3f}, B={out_b:.3f}, Mean={out_m:.3f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
