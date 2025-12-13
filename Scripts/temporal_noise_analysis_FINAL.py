"""
完全版 - get_spectrumの実装に基づいた正確な再現

重要な発見：
1. get_spectrum()は crop_array() している！
2. カメラゲインの単位が逆の可能性が高い
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Tuple
import tifffile
from PIL import Image
from skimage.restoration import unwrap_phase
from scipy.stats import linregress
from qpi import QPIParameters, get_field, make_disk, crop_array


# =============================================================================
# あなたのget_spectrumと完全に一貫した実装
# =============================================================================

def extract_alpha_beta_using_your_code(
    hologram: np.ndarray,
    params: QPIParameters
) -> Tuple[np.ndarray, np.ndarray]:
    """
    あなたのget_field, get_spectrumと完全に同じ方法でα, βを抽出
    
    重要：
    - FFT → マスク → Crop → IFFT の順序
    - これはあなたのvisibility関数と同じプロセス
    """
    M, N = params.img_shape
    radius = params.aperturesize // 2
    
    # FFT
    array_fft = np.fft.fftshift(np.fft.fft2(hologram))
    
    # === DC成分（0次光）の抽出 ===
    # マスク適用
    mask_0th = make_disk(params.img_center, radius, params.img_shape)
    array_fft_0th = array_fft * mask_0th
    
    # Crop（get_spectrumと同じ）
    array_fft_0th_cropped = crop_array(
        array_fft_0th, 
        params.img_center, 
        params.aperturesize
    )
    
    # IFFT
    dc_field = np.fft.ifft2(np.fft.ifftshift(array_fft_0th_cropped))
    alpha = np.abs(dc_field)
    
    # === サイドバンド（1次光）の抽出 ===
    # これはget_field()が実際にやっていること
    field = get_field(hologram, params)
    beta = np.abs(field)
    
    return alpha, beta


def calculate_ALG_sensitivity_FINAL(
    hologram: np.ndarray,
    params: QPIParameters,
    camera_gain: float,
    check_gain_unit: bool = True
) -> np.ndarray:
    """
    論文Eq. (12)に基づくALG感度計算【最終版】
    
    Eq. (12): σ_φ = sqrt(S*α / (2*g*M*N*β²))
    
    Args:
        hologram: ホログラム画像
        params: QPIパラメータ
        camera_gain: カメラゲイン（e-/ADU）
        check_gain_unit: Trueの場合、ゲインの妥当性をチェック
    
    重要な発見：
    - get_field()は実際にはcrop_array()している
    - そのため、α, βはcropされた配列のサイズに依存
    - カメラゲインの単位に注意（e-/ADU）
    """
    # カメラゲインの妥当性チェック
    if check_gain_unit and camera_gain < 0.1:
        print(f"\n⚠️  WARNING: Camera gain {camera_gain:.4f} e-/ADU is unusually low!")
        print(f"    Typical range: 0.1 - 10 e-/ADU")
        print(f"    Possible issue: Unit might be inverted (ADU/e- instead of e-/ADU)")
        print(f"    Suggested value: {1/camera_gain:.2f} e-/ADU")
        print(f"\n    Continuing with original value, but please verify!")
    
    alpha, beta = extract_alpha_beta_using_your_code(hologram, params)
    
    # フィルタ開口面積 S
    filter_radius = params.aperturesize // 2
    S = np.pi * filter_radius ** 2
    
    M, N = params.img_shape
    
    # Eq. (12): σ_φ = sqrt(S*α / (2*g*M*N*β²))
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_phi = np.sqrt(S * alpha / (2 * camera_gain * M * N * beta**2))
        sigma_phi[~np.isfinite(sigma_phi)] = 0
    
    return sigma_phi


def verify_camera_gain(gain_results: dict, verbose: bool = True) -> dict:
    """
    カメラゲインの妥当性を詳細チェック
    
    Returns:
        dict with 'original_gain', 'inverted_gain', 'recommended_gain'
    """
    gain = gain_results['gain']
    inverted_gain = 1 / gain if gain > 0 else 0
    
    if verbose:
        print("\n" + "="*70)
        print("CAMERA GAIN VALIDATION")
        print("="*70)
        
        print(f"\n[Measured Values]")
        print(f"  Original gain:      {gain:.4f} e-/ADU")
        print(f"  Inverted (1/gain):  {inverted_gain:.2f} e-/ADU")
        print(f"  Read noise:         {gain_results['readnoise_electrons']:.2f} e-")
        print(f"  R² (fit quality):   {gain_results['r_squared']:.4f}")
        
        print(f"\n[Typical Camera Specifications]")
        print(f"  Basler cameras:     0.1 - 2.0 e-/ADU")
        print(f"  Scientific CMOS:    0.5 - 5.0 e-/ADU")
        print(f"  CCD cameras:        1.0 - 10.0 e-/ADU")
        
        print(f"\n[Assessment]")
        if 0.1 <= gain <= 10:
            print(f"  ✓ Original gain ({gain:.4f}) is in typical range")
            recommended = gain
            print(f"  → Recommended: use {recommended:.4f} e-/ADU")
        elif 0.1 <= inverted_gain <= 10:
            print(f"  ⚠️  Original gain ({gain:.4f}) is too low!")
            print(f"  ✓ Inverted gain ({inverted_gain:.2f}) is in typical range")
            recommended = inverted_gain
            print(f"  → Recommended: use {recommended:.2f} e-/ADU (inverted)")
        else:
            print(f"  ✗ Both values are outside typical range!")
            print(f"  → Please check the gain estimation method")
            recommended = gain  # デフォルトで元の値
        
        print("="*70)
    else:
        # 自動判定
        if 0.1 <= gain <= 10:
            recommended = gain
        elif 0.1 <= inverted_gain <= 10:
            recommended = inverted_gain
        else:
            recommended = gain
    
    return {
        'original_gain': gain,
        'inverted_gain': inverted_gain,
        'recommended_gain': recommended,
        'use_inverted': (0.1 <= inverted_gain <= 10) and not (0.1 <= gain <= 10)
    }


# =============================================================================
# その他の関数（前回と同じ）
# =============================================================================

def load_hologram_sequence(
    folder_path: Union[str, Path],
    n_frames: int = None,
    crop_region: Tuple[int, int, int, int] = None
) -> np.ndarray:
    """連続したホログラムを読み込む"""
    folder = Path(folder_path)
    
    tif_files = sorted(folder.glob("*.tif")) + sorted(folder.glob("*.tiff"))
    tif_files = sorted(tif_files)
    tif_files = [f for f in tif_files if not f.name.startswith('.')]
    
    print(f"Folder: {folder}")
    print(f"Found {len(tif_files)} image files")
    
    if len(tif_files) == 0:
        print("ERROR: No .tif or .tiff files found!")
        return np.array([])
    
    if n_frames is not None:
        tif_files = tif_files[:n_frames]
    
    holograms = []
    for i, file in enumerate(tif_files):
        if i % 100 == 0:
            print(f"  Loading {i}/{len(tif_files)}")
        
        try:
            img = tifffile.imread(file)
            if len(img.shape) == 3:
                img = img[:, :, 0]
            if crop_region is not None:
                y1, y2, x1, x2 = crop_region
                img = img[y1:y2, x1:x2]
            holograms.append(img)
        except Exception as e:
            print(f"    ✗ Failed to load {file.name}: {e}")
            continue
    
    print(f"✓ Successfully loaded {len(holograms)} images")
    return np.array(holograms)


def estimate_camera_gain_from_temporal_variance(
    holograms: np.ndarray,
    n_regions: int = 50,
    region_size: int = 30
) -> dict:
    """時間的な平均-分散関係からカメラゲインを推定"""
    n_frames, H, W = holograms.shape
    
    means = []
    variances = []
    
    print(f"Sampling {n_regions} regions for gain estimation...")
    
    for i in range(n_regions):
        y = np.random.randint(50, H - region_size - 50)
        x = np.random.randint(50, W - region_size - 50)
        
        region_sequence = holograms[:, y:y+region_size, x:x+region_size]
        
        temporal_mean = np.mean(region_sequence)
        temporal_var = np.var(region_sequence)
        
        means.append(temporal_mean)
        variances.append(temporal_var)
    
    means = np.array(means)
    variances = np.array(variances)
    
    slope, intercept, r_value, p_value, std_err = linregress(means, variances)
    
    gain_estimate = 1 / slope
    readnoise_estimate = np.sqrt(max(intercept, 0))
    
    return {
        'means': means,
        'variances': variances,
        'gain': gain_estimate,
        'readnoise_ADU': readnoise_estimate,
        'readnoise_electrons': readnoise_estimate * gain_estimate,
        'r_squared': r_value**2
    }


def calculate_EXP_sensitivity_with_bg_frame(
    holograms: np.ndarray,
    bg_hologram: np.ndarray,
    params: QPIParameters,
    use_unwrap: bool = True,
    bg_region: Tuple[int, int, int, int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """バックグラウンド差分による実験的感度測定"""
    n_frames = holograms.shape[0]
    phases_diff = []
    
    print("Calculating background phase...")
    field_bg = get_field(bg_hologram, params)
    phase_bg = np.angle(field_bg)
    
    if use_unwrap:
        phase_bg = unwrap_phase(phase_bg)
    
    print(f"\nProcessing {n_frames} frames...")
    for i in range(n_frames):
        if i % 100 == 0:
            print(f"  Frame {i}/{n_frames}")
        
        field = get_field(holograms[i], params)
        phase = np.angle(field)
        
        if use_unwrap:
            phase = unwrap_phase(phase)
        
        phase_diff = phase - phase_bg
        
        if bg_region is not None:
            y1, y2, x1, x2 = bg_region
            offset = np.mean(phase_diff[y1:y2, x1:x2])
            phase_diff = phase_diff - offset
        
        phases_diff.append(phase_diff)
    
    phases_diff = np.array(phases_diff)
    sigma_exp = np.std(phases_diff, axis=0)
    
    return sigma_exp, phases_diff


def plot_fig3_style(
    sigma_exp: np.ndarray,
    sigma_alg: np.ndarray,
    wavelength: float,
    vmax_factor: float = 3.0,
    save_path: str = None
):
    """論文 Fig. 3 のスタイルでプロット"""
    mean_exp = np.mean(sigma_exp[sigma_exp > 0])
    mean_alg = np.mean(sigma_alg[sigma_alg > 0])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        efficiency = sigma_alg / sigma_exp * 100
        efficiency[~np.isfinite(efficiency)] = 0
        efficiency = np.clip(efficiency, 0, 150)
    mean_eff = np.mean(efficiency[(efficiency > 0) & (efficiency < 150)])
    
    vmax = mean_exp * vmax_factor
    
    fig = plt.figure(figsize=(12, 4))
    
    # (a) EXP
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(sigma_exp, cmap='hot', vmin=0, vmax=vmax)
    ax1.set_title(f'(a) EXP\nmean: {mean_exp:.4f} rad', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Pixel', fontsize=10)
    ax1.set_ylabel('Pixel', fontsize=10)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='σ_φ (rad)')
    
    # (b) ALG
    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.imshow(sigma_alg, cmap='hot', vmin=0, vmax=vmax)
    ax2.set_title(f'(b) ALG\nmean: {mean_alg:.4f} rad', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Pixel', fontsize=10)
    ax2.set_ylabel('Pixel', fontsize=10)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='σ_φ (rad)')
    
    # (c) 中央列の比較
    ax3 = plt.subplot(1, 3, 3)
    center_col = sigma_exp.shape[1] // 2
    y_pixels = np.arange(sigma_exp.shape[0])
    
    ax3.plot(y_pixels, sigma_exp[:, center_col], 'b-', 
             label='EXP', linewidth=2, alpha=0.8)
    ax3.plot(y_pixels, sigma_alg[:, center_col], 'r--', 
             label='ALG', linewidth=2, alpha=0.8)
    
    ax3.set_xlabel('Pixel', fontsize=10)
    ax3.set_ylabel('σ_φ (rad)', fontsize=10)
    ax3.set_title(f'(c) Center column\nEfficiency: {mean_eff:.1f}%', 
                  fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9, loc='upper right')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim([0, sigma_exp.shape[0]])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS RESULTS")
    print("="*60)
    print(f"Mean EXP sensitivity: {mean_exp:.6e} rad")
    print(f"Mean ALG sensitivity: {mean_alg:.6e} rad")
    print(f"System efficiency:    {mean_eff:.2f}%")
    print(f"ALG/EXP ratio:        {mean_alg/mean_exp:.4f}")
    
    # OPL単位に変換
    k0 = 2 * np.pi / wavelength
    mean_exp_opl = mean_exp / k0 * 1e9
    mean_alg_opl = mean_alg / k0 * 1e9
    print(f"\nOPL sensitivity:")
    print(f"  EXP: {mean_exp_opl:.2f} nm")
    print(f"  ALG: {mean_alg_opl:.2f} nm")
    print("="*60)
    
    # 妥当性チェック
    if mean_eff < 80:
        print("\n⚠️  WARNING: System efficiency < 80%")
        print("    Possible issues:")
        print("    - Camera gain value may be incorrect (check unit!)")
        print("    - System has significant instabilities")
    elif mean_eff > 110:
        print("\n⚠️  WARNING: System efficiency > 110%")
        print("    Possible issues:")
        print("    - Camera gain value may be incorrect (check unit!)")
        print("    - Too few frames for EXP (try 500+ frames)")
        print("    - α, β extraction may have issues")
    else:
        print("\n✓ System efficiency is within expected range (80-110%)")


def plot_gain_calibration(gain_results: dict, save_path: str = None):
    """カメラゲイン校正結果をプロット"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(gain_results['means'], gain_results['variances'], 
               alpha=0.6, s=50, label='Data')
    
    mean_range = np.array([gain_results['means'].min(), gain_results['means'].max()])
    fit_line = mean_range / gain_results['gain'] + gain_results['readnoise_ADU']**2
    ax.plot(mean_range, fit_line, 'r-', linewidth=2, 
            label=f"Fit: g={gain_results['gain']:.4f} e-/ADU")
    
    shot_noise_line = mean_range / gain_results['gain']
    ax.plot(mean_range, shot_noise_line, 'g--', linewidth=2, alpha=0.5,
            label='Shot noise limit')
    
    ax.set_xlabel('Mean intensity (ADU)', fontsize=12)
    ax.set_ylabel('Temporal variance (ADU²)', fontsize=12)
    ax.set_title('Mean-Variance Relationship for Gain Calibration', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()

# %%
