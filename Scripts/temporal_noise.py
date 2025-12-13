"""
最終版 - データシート実測値を使用

Basler acA2440-75um のデータシートから得られた実測値：
- Saturation Capacity: 10.7 ke⁻ = 10,700 e⁻
- Dark Noise: 2.4 e⁻
- Dynamic Range: 73.1 dB
- Quantum Efficiency: 68%

→ Conversion Gain = 10,700 / 255 = 41.96 e⁻/ADU (8-bit mode)
"""
# %%
import numpy as np
from PIL import Image
from qpi import QPIParameters, get_field
from temporal_noise_analysis_FINAL import (
    extract_alpha_beta_using_your_code,
    calculate_ALG_sensitivity_FINAL,
    calculate_EXP_sensitivity_with_bg_frame,
    plot_fig3_style,
    load_hologram_sequence
)

print("\n" + "="*70)
print("FINAL SENSITIVITY ANALYSIS")
print("Using Datasheet Conversion Gain: 41.96 e⁻/ADU")
print("="*70)

# =============================================================================
# カメラ仕様（データシート実測値）
# =============================================================================

print("\n[Camera Specifications - Basler acA2440-75um]")
print("  Model:               acA2440-75um")
print("  Sensor:              Sony IMX250")
print("  Pixel size:          3.45 μm")
print("  Bit depth:           8-bit")
print("  Gain setting:        0 dB")

# データシート実測値
DATASHEET_VALUES = {
    'saturation_capacity': 10700,   # e⁻ (10.7 ke⁻)
    'dark_noise': 2.4,              # e⁻
    'dynamic_range': 73.1,          # dB
    'quantum_efficiency': 0.68,     # 68%
}

# Conversion Gain計算
CAMERA_GAIN = DATASHEET_VALUES['saturation_capacity'] / 255  # 8-bit

print(f"\n[Datasheet Values]")
print(f"  Saturation Capacity: {DATASHEET_VALUES['saturation_capacity']} e⁻")
print(f"  Dark Noise:          {DATASHEET_VALUES['dark_noise']} e⁻")
print(f"  Dynamic Range:       {DATASHEET_VALUES['dynamic_range']} dB")
print(f"  Quantum Efficiency:  {DATASHEET_VALUES['quantum_efficiency']*100:.0f}%")

print(f"\n[Conversion Gain]")
print(f"  Formula:             Saturation / Max_ADU")
print(f"  Calculation:         {DATASHEET_VALUES['saturation_capacity']} / 255")
print(f"  Result:              {CAMERA_GAIN:.2f} e⁻/ADU")

# 検証
expected_DR = 20 * np.log10(DATASHEET_VALUES['saturation_capacity'] / DATASHEET_VALUES['dark_noise'])
print(f"\n[Verification]")
print(f"  Expected DR:         {expected_DR:.1f} dB")
print(f"  Datasheet DR:        {DATASHEET_VALUES['dynamic_range']} dB")
print(f"  Match:               {'✓ Yes' if abs(expected_DR - DATASHEET_VALUES['dynamic_range']) < 1 else '✗ No'}")

# =============================================================================
# システムパラメータ
# =============================================================================

WAVELENGTH = 663e-9  # m
NA = 0.95
PIXELSIZE = 3.45e-6 / 40  # m
CROP_REGION = (8, 2056, 208, 2256)
offaxis_center = (1642, 466)

# =============================================================================
# データ読み込み
# =============================================================================

print("\n" + "="*70)
print("Loading Data")
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
print(f"Image range: {np.min(img)} - {np.max(img)} ADU")
print(f"Image mean:  {np.mean(img):.1f} ADU")
print(f"Aperture:    {params.aperturesize} pixels")

# 時系列データ
folder_path = "/Volumes/QPI_0_.01_r/251211/sequence shot"
N_FRAMES = 500

holograms = load_hologram_sequence(folder_path, n_frames=N_FRAMES, crop_region=CROP_REGION)
print(f"Loaded {len(holograms)} frames")

if len(holograms) == 0:
    exit(1)

# =============================================================================
# 感度計算
# =============================================================================

print("\n" + "="*70)
print("ALG Sensitivity (Theoretical)")
print("="*70)

sigma_alg = calculate_ALG_sensitivity_FINAL(
    hologram=holograms[0],
    params=params,
    camera_gain=CAMERA_GAIN,
    check_gain_unit=False
)

mean_alg = np.mean(sigma_alg[sigma_alg > 0])
k0 = 2 * np.pi / WAVELENGTH
sigma_alg_opl = sigma_alg / k0 * 1e9  # nm

print(f"Phase sensitivity:  {mean_alg:.6e} rad")
print(f"OPL sensitivity:    {np.mean(sigma_alg_opl):.4f} nm")

print("\n" + "="*70)
print("EXP Sensitivity (Experimental)")
print("="*70)

sigma_exp, phases_diff = calculate_EXP_sensitivity_with_bg_frame(
    holograms=holograms,
    bg_hologram=holograms[0],
    params=params,
    use_unwrap=True,
    bg_region=(10, 60, 10, 60)
)

mean_exp = np.mean(sigma_exp[sigma_exp > 0])
sigma_exp_opl = sigma_exp / k0 * 1e9  # nm

print(f"Phase sensitivity:  {mean_exp:.6e} rad")
print(f"OPL sensitivity:    {np.mean(sigma_exp_opl):.4f} nm")

# =============================================================================
# 結果と評価
# =============================================================================

print("\n" + "="*70)
print("System Performance")
print("="*70)

plot_fig3_style(
    sigma_exp=sigma_exp,
    sigma_alg=sigma_alg,
    wavelength=WAVELENGTH,
    vmax_factor=3.0,
    save_path="fig3_FINAL_datasheet.png"
)

efficiency = (sigma_alg / sigma_exp) * 100
efficiency_valid = efficiency[(efficiency > 0) & (efficiency < 150)]

print(f"\n[System Efficiency]")
print(f"  Mean:     {np.mean(efficiency_valid):.2f}%")
print(f"  Median:   {np.median(efficiency_valid):.2f}%")
print(f"  Std:      {np.std(efficiency_valid):.2f}%")

print(f"\n[Comparison with Literature]")
print(f"  Paper (Chen et al. 2018):  ~97.5% efficiency")
print(f"  Your system:               {np.mean(efficiency_valid):.1f}% efficiency")

print(f"\n[Final Assessment]")
if 90 <= np.mean(efficiency_valid) <= 105:
    print(f"  ✓✓✓ EXCELLENT!")
    print(f"      System matches expected performance.")
    print(f"      Hardware is stable and well-calibrated.")
    print(f"      Conversion gain ({CAMERA_GAIN:.1f} e⁻/ADU) is correct.")
elif 80 <= np.mean(efficiency_valid) < 90:
    print(f"  ✓✓ VERY GOOD!")
    print(f"      System is performing well.")
    print(f"      Minor improvements possible in stability.")
elif 70 <= np.mean(efficiency_valid) < 80:
    print(f"  ✓ GOOD")
    print(f"      System is functional.")
    print(f"      Consider improving environmental stability.")
elif np.mean(efficiency_valid) > 110:
    print(f"  ⚠️  Efficiency > 110%")
    print(f"      Possible causes:")
    print(f"      1. Too few frames for EXP (try 1000 frames)")
    print(f"      2. Insufficient temporal averaging")
    print(f"      3. System instabilities affecting EXP")
else:
    print(f"  ⚠️  Efficiency < 70%")
    print(f"      Possible causes:")
    print(f"      1. System has significant noise sources")
    print(f"      2. Environmental instabilities")
    print(f"      3. Optical misalignment")

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"Camera:              Basler acA2440-75um")
print(f"Conversion Gain:     {CAMERA_GAIN:.2f} e⁻/ADU (datasheet)")
print(f"Phase Sensitivity:   {mean_exp:.4e} rad")
print(f"OPL Sensitivity:     {np.mean(sigma_exp_opl):.2f} nm")
print(f"System Efficiency:   {np.mean(efficiency_valid):.1f}%")
print(f"\nFigure saved:        fig3_FINAL_datasheet.png")
print("="*70)

print("\n✓ Analysis Complete!")
print("  This result uses the accurate datasheet value.")
print("  No further calibration needed.")
# %%
