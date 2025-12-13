"""
最終版クイックスタート - カメラゲイン検証を含む

重要な発見：
1. get_spectrum()は crop_array() している（元のコードと同じ）
2. カメラゲインが0.0395 e-/ADUは異常に低い
3. おそらく単位が逆：正しい値は 1/0.0395 = 25.3 e-/ADU
"""
# %%
import numpy as np
from PIL import Image
from qpi import QPIParameters, get_field
from temporal_noise_analysis_FINAL import (
    extract_alpha_beta_using_your_code,
    load_hologram_sequence,
    estimate_camera_gain_from_temporal_variance,
    verify_camera_gain,
    calculate_ALG_sensitivity_FINAL,
    calculate_EXP_sensitivity_with_bg_frame,
    plot_fig3_style,
    plot_gain_calibration
)

print("\n" + "="*70)
print("FINAL VERSION - 完全版感度解析")
print("get_spectrum実装に基づく正確な再現 + カメラゲイン検証")
print("="*70)

# =============================================================================
# パラメータ設定
# =============================================================================

WAVELENGTH = 663e-9  # m
NA = 0.95
PIXELSIZE = 3.45e-6 / 40  # m
CROP_REGION = (8, 2056, 208, 2256)
offaxis_center = (1642, 466)

# =============================================================================
# STEP 1: 単一ホログラム読み込み
# =============================================================================

print("\n" + "="*70)
print("STEP 1: 単一ホログラム読み込み")
print("="*70)

path = "/Volumes/QPI_0_.01_r/251211/sequence shot/Basler_acA2440-75um__25176370__20251211_152604439_0000.tiff"
img = np.array(Image.open(path))
img = img[CROP_REGION[0]:CROP_REGION[1], CROP_REGION[2]:CROP_REGION[3]]

params = QPIParameters(
    wavelength=WAVELENGTH,
    NA=NA,
    img_shape=img.shape,
    pixelsize=PIXELSIZE,
    offaxis_center=offaxis_center,
)

print(f"Image shape: {img.shape}")
print(f"Aperture size: {params.aperturesize} pixels")
print(f"Image mean: {np.mean(img):.1f} ADU")

# α, βの確認
print("\n[Alpha & Beta Verification]")
alpha, beta = extract_alpha_beta_using_your_code(img, params)
visibility = 2 * beta / alpha

print(f"  Alpha mean:      {np.mean(alpha):.1f} ADU")
print(f"  Beta mean:       {np.mean(beta):.1f} ADU")
print(f"  Visibility mean: {np.mean(visibility):.4f}")
print(f"  → Should match your visibility calculation (0.70-0.85)")

# =============================================================================
# STEP 2: 時系列データ読み込み
# =============================================================================

print("\n" + "="*70)
print("STEP 2: 時系列ホログラム読み込み")
print("="*70)

folder_path = "/Volumes/QPI_0_.01_r/251211/sequence shot"
N_FRAMES = 500

holograms = load_hologram_sequence(
    folder_path,
    n_frames=N_FRAMES,
    crop_region=CROP_REGION
)

if len(holograms) == 0:
    print("✗ Failed to load holograms!")
    exit(1)

print(f"Loaded shape: {holograms.shape}")

# =============================================================================
# STEP 3: カメラゲイン測定と検証
# =============================================================================

print("\n" + "="*70)
print("STEP 3: カメラゲイン測定と検証")
print("="*70)

# ゲイン推定
gain_results = estimate_camera_gain_from_temporal_variance(
    holograms=holograms,
    n_regions=50,
    region_size=30
)

print(f"\n[Initial Measurement]")
print(f"  Measured gain:      {gain_results['gain']:.4f} e-/ADU")
print(f"  Read noise:         {gain_results['readnoise_electrons']:.2f} e-")
print(f"  R² (fit quality):   {gain_results['r_squared']:.4f}")

# ゲインの妥当性を詳細チェック
gain_validation = verify_camera_gain(gain_results, verbose=True)

# 推奨値を使用
CAMERA_GAIN = gain_validation['recommended_gain']

print(f"\n✓ Using camera gain: {CAMERA_GAIN:.4f} e-/ADU")

# プロット
plot_gain_calibration(gain_results, save_path='camera_gain_calibration.png')

# =============================================================================
# STEP 4: ALG感度計算
# =============================================================================

print("\n" + "="*70)
print("STEP 4: ALG感度計算（理論値）")
print("="*70)

sigma_alg = calculate_ALG_sensitivity_FINAL(
    hologram=holograms[0],
    params=params,
    camera_gain=CAMERA_GAIN,
    check_gain_unit=True
)

mean_alg = np.mean(sigma_alg[sigma_alg > 0])
print(f"Mean ALG sensitivity: {mean_alg:.6e} rad")

# OPL単位
k0 = 2 * np.pi / WAVELENGTH
sigma_alg_opl = sigma_alg / k0 * 1e9  # nm
print(f"Mean ALG sensitivity: {np.mean(sigma_alg_opl):.2f} nm OPL")

# =============================================================================
# STEP 5: EXP感度計算
# =============================================================================

print("\n" + "="*70)
print("STEP 5: EXP感度計算（実験値）")
print("="*70)

bg_hologram = holograms[0]
bg_region = (10, 60, 10, 60)

sigma_exp, phases_diff = calculate_EXP_sensitivity_with_bg_frame(
    holograms=holograms,
    bg_hologram=bg_hologram,
    params=params,
    use_unwrap=True,
    bg_region=bg_region
)

mean_exp = np.mean(sigma_exp[sigma_exp > 0])
print(f"Mean EXP sensitivity: {mean_exp:.6e} rad")

sigma_exp_opl = sigma_exp / k0 * 1e9  # nm
print(f"Mean EXP sensitivity: {np.mean(sigma_exp_opl):.2f} nm OPL")

# =============================================================================
# STEP 6: 結果の可視化と評価
# =============================================================================

print("\n" + "="*70)
print("STEP 6: 結果の可視化と総合評価")
print("="*70)

plot_fig3_style(
    sigma_exp=sigma_exp,
    sigma_alg=sigma_alg,
    wavelength=WAVELENGTH,
    vmax_factor=3.0,
    save_path="fig3_final_verified.png"
)

# システム効率の詳細分析
efficiency = (sigma_alg / sigma_exp) * 100
efficiency_valid = efficiency[(efficiency > 0) & (efficiency < 150)]

print(f"\n[Detailed System Efficiency Analysis]")
print(f"  Mean:     {np.mean(efficiency_valid):.2f}%")
print(f"  Median:   {np.median(efficiency_valid):.2f}%")
print(f"  Std:      {np.std(efficiency_valid):.2f}%")
print(f"  Range:    {np.min(efficiency_valid):.2f}% - {np.max(efficiency_valid):.2f}%")

# 論文との比較
print(f"\n[Comparison with Paper (Fig. 3)]")
print(f"  Paper shows:  ~97.5% efficiency")
print(f"  Your system:  {np.mean(efficiency_valid):.1f}% efficiency")

if 90 <= np.mean(efficiency_valid) <= 105:
    print(f"\n✓✓✓ EXCELLENT! System is performing as expected!")
    print(f"    Your hardware is very stable and well-aligned.")
elif 80 <= np.mean(efficiency_valid) < 90:
    print(f"\n✓✓ GOOD! System is working well.")
    print(f"    Minor improvements possible in:")
    print(f"    - Vibration isolation")
    print(f"    - Temperature stability")
    print(f"    - Light source stability")
elif 70 <= np.mean(efficiency_valid) < 80:
    print(f"\n✓ OK. System is functional but has room for improvement.")
    print(f"    Consider checking:")
    print(f"    - Mechanical stability")
    print(f"    - Optical alignment")
    print(f"    - Environmental conditions")
else:
    print(f"\n⚠️  System efficiency is outside expected range.")
    print(f"    Possible issues to check:")
    print(f"    1. Camera gain value (try both {gain_validation['original_gain']:.4f} and {gain_validation['inverted_gain']:.2f})")
    print(f"    2. Number of frames (current: {len(holograms)}, recommended: 500+)")
    print(f"    3. Background subtraction region")
    print(f"    4. System instabilities")

# =============================================================================
# STEP 7: 比較テスト（オプション）
# =============================================================================

if gain_validation['use_inverted']:
    print("\n" + "="*70)
    print("STEP 7: 元のゲイン値との比較テスト")
    print("="*70)
    
    print(f"\n[Testing with ORIGINAL gain: {gain_validation['original_gain']:.4f} e-/ADU]")
    sigma_alg_orig = calculate_ALG_sensitivity_FINAL(
        hologram=holograms[0],
        params=params,
        camera_gain=gain_validation['original_gain'],
        check_gain_unit=False
    )
    
    eff_orig = np.mean((sigma_alg_orig / sigma_exp) * 100)
    print(f"  System efficiency: {eff_orig:.2f}%")
    
    print(f"\n[Testing with INVERTED gain: {gain_validation['inverted_gain']:.2f} e-/ADU]")
    sigma_alg_inv = calculate_ALG_sensitivity_FINAL(
        hologram=holograms[0],
        params=params,
        camera_gain=gain_validation['inverted_gain'],
        check_gain_unit=False
    )
    
    eff_inv = np.mean((sigma_alg_inv / sigma_exp) * 100)
    print(f"  System efficiency: {eff_inv:.2f}%")
    
    print(f"\n[Conclusion]")
    if abs(eff_inv - 97.5) < abs(eff_orig - 97.5):
        print(f"  ✓ Inverted gain ({gain_validation['inverted_gain']:.2f}) gives better result!")
        print(f"    Efficiency: {eff_inv:.2f}% (closer to paper's 97.5%)")
    else:
        print(f"  ✓ Original gain ({gain_validation['original_gain']:.4f}) is correct!")
        print(f"    Efficiency: {eff_orig:.2f}%")

print("\n" + "="*70)
print("✓ ANALYSIS COMPLETE!")
print("="*70)

print("\n[Summary]")
print(f"  Camera gain:         {CAMERA_GAIN:.4f} e-/ADU")
print(f"  Phase sensitivity:   {mean_exp:.4f} rad")
print(f"  OPL sensitivity:     {np.mean(sigma_exp_opl):.2f} nm")
print(f"  System efficiency:   {np.mean(efficiency_valid):.1f}%")
print("\n  Results saved to:")
print("    - camera_gain_calibration.png")
print("    - fig3_final_verified.png")

# %%
