# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Tuple
import tifffile
from PIL import Image
from skimage.restoration import unwrap_phase

# 既存のqpiモジュールからインポート
from qpi import QPIParameters, get_field, get_spectrum, make_disk, crop_array


# =============================================================================
# Temporal Noise Analysis用の新規関数
# =============================================================================

def load_hologram_sequence(
    folder_path: Union[str, Path],
    n_frames: int = None,
    crop_region: Tuple[int, int, int, int] = None
) -> np.ndarray:
    """
    連続したホログラムを読み込む
    
    Args:
        folder_path: ホログラムが保存されているフォルダパス
        n_frames: 読み込むフレーム数（Noneの場合は全フレーム）
        crop_region: (y_start, y_end, x_start, x_end) のクロップ領域
    
    Returns:
        holograms: shape (n_frames, height, width)
    """
    folder = Path(folder_path)
    tif_files = sorted(folder.glob("*.tif"))
    
    if n_frames is not None:
        tif_files = tif_files[:n_frames]
    
    holograms = []
    for file in tif_files:
        img = np.array(Image.open(file))
        
        if crop_region is not None:
            y1, y2, x1, x2 = crop_region
            img = img[y1:y2, x1:x2]
        
        holograms.append(img)
    
    return np.array(holograms)


def extract_alpha_beta(
    hologram: np.ndarray,
    params: QPIParameters
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ホログラムからα (DC成分) とβ (干渉縞振幅) を抽出
    論文のEq. (1)に対応
    
    Args:
        hologram: 入力ホログラム
        params: QPIパラメータ
    
    Returns:
        alpha: DC強度分布
        beta: 干渉縞振幅分布
    """
    # FFT
    fft_holo = np.fft.fftshift(np.fft.fft2(hologram))
    
    # DC成分 (0次光) の抽出
    dc_mask = make_disk(params.img_center, params.aperturesize // 2, params.img_shape)
    dc_fft = fft_holo * dc_mask
    dc_cropped = crop_array(dc_fft, params.img_center, params.aperturesize)
    dc_field = np.fft.ifft2(np.fft.ifftshift(dc_cropped))
    
    # スケーリング係数
    scale_factor = params.aperturesize / params.img_shape[0]
    alpha = np.abs(dc_field) * scale_factor**2
    
    # サイドバンド成分 (1次光) の抽出
    sb_mask = make_disk(params.offaxis_center, params.aperturesize // 2, params.img_shape)
    sb_fft = fft_holo * sb_mask
    sb_cropped = crop_array(sb_fft, params.offaxis_center, params.aperturesize)
    sb_field = np.fft.ifft2(np.fft.ifftshift(sb_cropped))
    beta = np.abs(sb_field) * scale_factor**2
    
    return alpha, beta


def calculate_ALG_sensitivity_shot_noise(
    hologram: np.ndarray,
    params: QPIParameters,
    camera_gain: float,
    filter_bandwidth_ratio: float = 0.3
) -> np.ndarray:
    """
    論文のEq. (12)に基づくALG感度計算（ショットノイズモデル）
    
    Args:
        hologram: 単一ホログラム
        params: QPIパラメータ
        camera_gain: カメラゲイン [e-/ADU]
        filter_bandwidth_ratio: フィルタ帯域幅の比率
    
    Returns:
        sigma_phi: 位相感度マップ [rad]
    """
    # α, βの抽出
    alpha, beta = extract_alpha_beta(hologram, params)
    
    # フィルタ開口面積 S の計算
    radius = filter_bandwidth_ratio * np.sqrt(
        (params.offaxis_center[0] - params.img_center[0])**2 + 
        (params.offaxis_center[1] - params.img_center[1])**2
    )
    S = np.pi * radius**2
    
    # センサー全体のピクセル数
    M, N = params.img_shape
    
    # Eq. (12): σ_φ = sqrt(S*α / (2*g*M*N*β²))
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_phi = np.sqrt(S * alpha / (2 * camera_gain * M * N * beta**2))
        sigma_phi[~np.isfinite(sigma_phi)] = 0
    
    return sigma_phi


def calculate_EXP_sensitivity(
    holograms: np.ndarray,
    params: QPIParameters,
    use_unwrap: bool = True
) -> np.ndarray:
    """
    時系列ホログラムから実験的位相感度 (EXP) を計算
    
    Args:
        holograms: shape (n_frames, height, width)
        params: QPIパラメータ
        use_unwrap: 位相アンラップを使用するか
    
    Returns:
        sigma_exp: 実験的位相感度 [rad]
    """
    n_frames = holograms.shape[0]
    phases = []
    
    print(f"Processing {n_frames} frames...")
    for i in range(n_frames):
        if i % 100 == 0:
            print(f"  Frame {i}/{n_frames}")
        
        # 位相再構成
        field = get_field(holograms[i], params)
        phase = np.angle(field)
        
        if use_unwrap:
            phase = unwrap_phase(phase)
        
        phases.append(phase)
    
    phases = np.array(phases)
    
    # 時間方向の標準偏差
    sigma_exp = np.std(phases, axis=0)
    
    return sigma_exp


def calculate_EXP_sensitivity_differential(
    holograms: np.ndarray,
    holograms_bg: np.ndarray,
    params: QPIParameters,
    use_unwrap: bool = True,
    bg_region: Tuple[int, int, int, int] = None
) -> np.ndarray:
    """
    バックグラウンド差分を取った後の実験的位相感度 (EXP) を計算
    
    Args:
        holograms: サンプルホログラム系列 shape (n_frames, height, width)
        holograms_bg: バックグラウンドホログラム系列
        params: QPIパラメータ
        use_unwrap: 位相アンラップを使用するか
        bg_region: バックグラウンド補正用の領域 (y1, y2, x1, x2)
    
    Returns:
        sigma_exp: 実験的位相感度 [rad]
    """
    assert holograms.shape == holograms_bg.shape
    n_frames = holograms.shape[0]
    phases_diff = []
    
    print(f"Processing {n_frames} frames with background subtraction...")
    for i in range(n_frames):
        if i % 100 == 0:
            print(f"  Frame {i}/{n_frames}")
        
        # サンプルとバックグラウンドの位相再構成
        field = get_field(holograms[i], params)
        field_bg = get_field(holograms_bg[i], params)
        
        phase = np.angle(field)
        phase_bg = np.angle(field_bg)
        
        if use_unwrap:
            phase = unwrap_phase(phase)
            phase_bg = unwrap_phase(phase_bg)
        
        # 差分位相
        phase_diff = phase - phase_bg
        
        # バックグラウンド領域で補正
        if bg_region is not None:
            y1, y2, x1, x2 = bg_region
            offset = np.mean(phase_diff[y1:y2, x1:x2])
            phase_diff = phase_diff - offset
        
        phases_diff.append(phase_diff)
    
    phases_diff = np.array(phases_diff)
    
    # 時間方向の標準偏差
    sigma_exp = np.std(phases_diff, axis=0)
    
    return sigma_exp


def estimate_camera_gain(
    hologram: np.ndarray,
    n_samples: int = 100
) -> float:
    """
    論文のEq. (9)を使ってカメラゲインを推定
    mean-variance関係から g = mean / variance
    
    Args:
        hologram: ホログラム（複数フレームの平均でも可）
        n_samples: サンプリング数
    
    Returns:
        camera_gain: 推定されたカメラゲイン [e-/ADU]
    """
    # ランダムなパッチを抽出して平均と分散を計算
    H, W = hologram.shape
    patch_size = 20
    
    means = []
    variances = []
    
    for _ in range(n_samples):
        y = np.random.randint(0, H - patch_size)
        x = np.random.randint(0, W - patch_size)
        patch = hologram[y:y+patch_size, x:x+patch_size]
        
        means.append(np.mean(patch))
        variances.append(np.var(patch))
    
    means = np.array(means)
    variances = np.array(variances)
    
    # 線形フィッティング: variance = mean / g
    # g = mean / variance
    gain = np.mean(means / variances)
    
    return gain


def plot_sensitivity_comparison(
    sigma_exp: np.ndarray,
    sigma_alg: np.ndarray,
    wavelength: float,
    save_path: str = None
):
    """
    EXPとALGの比較プロット（論文 Fig. 3に相当）
    
    Args:
        sigma_exp: 実験的感度
        sigma_alg: アルゴリズム感度
        wavelength: 波長 [m]
        save_path: 保存パス（Noneの場合は保存しない）
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 位相感度からOPL感度への変換
    k0 = 2 * np.pi / wavelength
    sigma_exp_opl = sigma_exp / k0 * 1e9  # nm
    sigma_alg_opl = sigma_alg / k0 * 1e9  # nm
    
    # (a) EXP sensitivity
    im0 = axes[0, 0].imshow(sigma_exp, cmap='hot', vmin=0)
    axes[0, 0].set_title('(a) Experimental Sensitivity (EXP)')
    axes[0, 0].set_xlabel('Pixel X')
    axes[0, 0].set_ylabel('Pixel Y')
    plt.colorbar(im0, ax=axes[0, 0], label='σ_φ (rad)')
    
    # (b) ALG sensitivity
    im1 = axes[0, 1].imshow(sigma_alg, cmap='hot', vmin=0)
    axes[0, 1].set_title('(b) Algorithm Sensitivity (ALG)')
    axes[0, 1].set_xlabel('Pixel X')
    axes[0, 1].set_ylabel('Pixel Y')
    plt.colorbar(im1, ax=axes[0, 1], label='σ_φ (rad)')
    
    # (c) Line profile comparison
    center_row = sigma_exp.shape[0] // 2
    axes[0, 2].plot(sigma_exp[center_row, :], 'b-', label='EXP', linewidth=2)
    axes[0, 2].plot(sigma_alg[center_row, :], 'r--', label='ALG', linewidth=2)
    axes[0, 2].set_xlabel('Pixel X')
    axes[0, 2].set_ylabel('σ_φ (rad)')
    axes[0, 2].set_title('(c) Center Row Profile')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # (d) EXP sensitivity in OPL
    im3 = axes[1, 0].imshow(sigma_exp_opl, cmap='hot', vmin=0)
    axes[1, 0].set_title('(d) EXP Sensitivity (OPL)')
    axes[1, 0].set_xlabel('Pixel X')
    axes[1, 0].set_ylabel('Pixel Y')
    plt.colorbar(im3, ax=axes[1, 0], label='σ_L (nm)')
    
    # (e) ALG sensitivity in OPL
    im4 = axes[1, 1].imshow(sigma_alg_opl, cmap='hot', vmin=0)
    axes[1, 1].set_title('(e) ALG Sensitivity (OPL)')
    axes[1, 1].set_xlabel('Pixel X')
    axes[1, 1].set_ylabel('Pixel Y')
    plt.colorbar(im4, ax=axes[1, 1], label='σ_L (nm)')
    
    # (f) System efficiency
    with np.errstate(divide='ignore', invalid='ignore'):
        efficiency = sigma_alg / sigma_exp * 100
        efficiency[~np.isfinite(efficiency)] = 0
        efficiency = np.clip(efficiency, 0, 100)
    
    mean_eff = np.mean(efficiency[efficiency > 0])
    
    im5 = axes[1, 2].imshow(efficiency, cmap='RdYlGn', vmin=80, vmax=100)
    axes[1, 2].set_title(f'(f) System Efficiency (mean: {mean_eff:.1f}%)')
    axes[1, 2].set_xlabel('Pixel X')
    axes[1, 2].set_ylabel('Pixel Y')
    plt.colorbar(im5, ax=axes[1, 2], label='Efficiency (%)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    
    # 統計情報の表示
    print("\n=== Sensitivity Statistics ===")
    print(f"EXP - Mean: {np.mean(sigma_exp):.4e} rad, Std: {np.std(sigma_exp):.4e} rad")
    print(f"ALG - Mean: {np.mean(sigma_alg):.4e} rad, Std: {np.std(sigma_alg):.4e} rad")
    print(f"System Efficiency: {mean_eff:.2f}%")
    print(f"\nOPL Sensitivity:")
    print(f"EXP - Mean: {np.mean(sigma_exp_opl):.2f} nm")
    print(f"ALG - Mean: {np.mean(sigma_alg_opl):.2f} nm")


# =============================================================================
# Fig. 3 風のプロット作成
# =============================================================================

def plot_fig3_style(
    sigma_exp: np.ndarray,
    sigma_alg: np.ndarray,
    wavelength: float,
    vmax_factor: float = 3.0,
    save_path: str = None
):
    """
    論文 Fig. 3 のスタイルでプロット
    (a) EXP, (b) ALG, (c) 中央列の比較, (d)-(f) は細胞の例
    
    Args:
        sigma_exp: 実験的感度
        sigma_alg: アルゴリズム感度  
        wavelength: 波長 [m]
        vmax_factor: カラーバーの最大値の倍率
        save_path: 保存パス
    """
    # 統計情報
    mean_exp = np.mean(sigma_exp[sigma_exp > 0])
    mean_alg = np.mean(sigma_alg[sigma_alg > 0])
    
    # システム効率
    with np.errstate(divide='ignore', invalid='ignore'):
        efficiency = sigma_alg / sigma_exp * 100
        efficiency[~np.isfinite(efficiency)] = 0
        efficiency = np.clip(efficiency, 0, 100)
    mean_eff = np.mean(efficiency[efficiency > 0])
    
    # カラーバーの範囲設定
    vmax = mean_exp * vmax_factor
    
    fig = plt.figure(figsize=(12, 4))
    
    # (a) EXP
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(sigma_exp, cmap='hot', vmin=0, vmax=vmax)
    ax1.set_title('(a) EXP', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Pixel', fontsize=10)
    ax1.set_ylabel('Pixel', fontsize=10)
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('σ_φ (rad)', fontsize=9)
    
    # (b) ALG
    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.imshow(sigma_alg, cmap='hot', vmin=0, vmax=vmax)
    ax2.set_title('(b) ALG', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Pixel', fontsize=10)
    ax2.set_ylabel('Pixel', fontsize=10)
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('σ_φ (rad)', fontsize=9)
    
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
    ax3.set_title('(c) Center column', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9, loc='upper right')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim([0, sigma_exp.shape[0]])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    # 統計情報の出力
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS RESULTS (Fig. 3 style)")
    print("="*60)
    print(f"Mean EXP sensitivity: {mean_exp:.6e} rad")
    print(f"Mean ALG sensitivity: {mean_alg:.6e} rad")
    print(f"System efficiency:    {mean_eff:.2f}%")
    print("="*60)


def plot_fig3_with_sample(
    sigma_exp_blank: np.ndarray,
    sigma_alg_blank: np.ndarray,
    sigma_alg_sample: np.ndarray,
    intensity_sample: np.ndarray,
    phase_sample: np.ndarray,
    wavelength: float,
    save_path: str = None
):
    """
    論文 Fig. 3 完全版（ブランクとサンプルの両方）
    
    Args:
        sigma_exp_blank: ブランクの実験的感度
        sigma_alg_blank: ブランクのアルゴリズム感度
        sigma_alg_sample: サンプルのアルゴリズム感度
        intensity_sample: サンプルの強度画像
        phase_sample: サンプルの位相画像
        wavelength: 波長 [m]
        save_path: 保存パス
    """
    fig = plt.figure(figsize=(12, 8))
    
    # カラーバーの範囲設定
    mean_exp = np.mean(sigma_exp_blank[sigma_exp_blank > 0])
    vmax_sensitivity = mean_exp * 3
    
    # (a) EXP - ブランク
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(sigma_exp_blank, cmap='hot', vmin=0, vmax=vmax_sensitivity)
    ax1.set_title('(a) EXP', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Pixel', fontsize=9)
    ax1.set_ylabel('Pixel', fontsize=9)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='σ_φ (rad)')
    
    # (b) ALG - ブランク
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(sigma_alg_blank, cmap='hot', vmin=0, vmax=vmax_sensitivity)
    ax2.set_title('(b) ALG', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Pixel', fontsize=9)
    ax2.set_ylabel('Pixel', fontsize=9)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='σ_φ (rad)')
    
    # (c) 中央列の比較
    ax3 = plt.subplot(2, 3, 3)
    center_col = sigma_exp_blank.shape[1] // 2
    y_pixels = np.arange(sigma_exp_blank.shape[0])
    
    ax3.plot(y_pixels, sigma_exp_blank[:, center_col], 'b-', 
             label='EXP', linewidth=2)
    ax3.plot(y_pixels, sigma_alg_blank[:, center_col], 'r--', 
             label='ALG', linewidth=2)
    ax3.set_xlabel('Pixel', fontsize=9)
    ax3.set_ylabel('σ_φ (rad)', fontsize=9)
    ax3.set_title('(c) Center column', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # (d) サンプル強度
    ax4 = plt.subplot(2, 3, 4)
    im4 = ax4.imshow(intensity_sample, cmap='gray')
    ax4.set_title('(d) Sample intensity', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Pixel', fontsize=9)
    ax4.set_ylabel('Pixel', fontsize=9)
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    # (e) サンプル位相
    ax5 = plt.subplot(2, 3, 5)
    im5 = ax5.imshow(phase_sample, cmap='gray')
    ax5.set_title('(e) Sample phase', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Pixel', fontsize=9)
    ax5.set_ylabel('Pixel', fontsize=9)
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04, label='Phase (rad)')
    
    # (f) サンプルのALG感度
    ax6 = plt.subplot(2, 3, 6)
    im6 = ax6.imshow(sigma_alg_sample, cmap='hot', vmin=0, vmax=vmax_sensitivity)
    ax6.set_title('(f) ALG from sample', fontsize=11, fontweight='bold')
    ax6.set_xlabel('Pixel', fontsize=9)
    ax6.set_ylabel('Pixel', fontsize=9)
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04, label='σ_φ (rad)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Complete figure saved to {save_path}")
    
    plt.show()


# =============================================================================
# 実行例: Fig. 3 スタイルの図を作成
# =============================================================================

if __name__ == "__main__":
    """
    Fig. 3を再現するために必要なもの:
    
    1. ブランク（ガラススライドのみ）の時系列ホログラム（800フレーム程度）
       → sigma_exp と sigma_alg を計算
    
    2. （オプション）サンプル（細胞など）の単一ホログラム
       → sigma_alg_sample を計算
    """
    
    # ========== パラメータ設定 ==========
    WAVELENGTH = 663e-9  # m
    NA = 0.95
    PIXELSIZE = 3.45e-6 / 40  # m
    CAMERA_GAIN = 34.4  # e-/ADU（要測定）
    
    # クロップ領域（あなたのコードと同じ）
    CROP_REGION = (8, 2056, 208, 2256)
    
    # ========== Step 1: ブランクの単一ホログラム読み込み ==========
    path_blank = "/Volumes/QPI_0_.01_r/251126_kk/ph_6/Pos1/img_000000000_Default_000.tif"
    
    img_blank = np.array(Image.open(path_blank))
    img_blank = img_blank[CROP_REGION[0]:CROP_REGION[1], 
                          CROP_REGION[2]:CROP_REGION[3]]
    
    # FFT確認（初回のみ）
    img_fft = np.fft.fftshift(np.fft.fft2(img_blank))
    plt.figure(figsize=(8, 6))
    plt.imshow(np.log(np.abs(img_fft)), cmap='hot')
    plt.title("FFT - Confirm off-axis center")
    plt.colorbar()
    plt.show()
    
    # off-axis centerを設定（FFTのピーク位置）
    offaxis_center = (1661, 486)  # ← 要調整
    
    # パラメータ設定
    params = QPIParameters(
        wavelength=WAVELENGTH,
        NA=NA,
        img_shape=img_blank.shape,
        pixelsize=PIXELSIZE,
        offaxis_center=offaxis_center,
    )
    
    print("\n=== QPI Parameters ===")
    print(f"Image shape: {params.img_shape}")
    print(f"Aperture size: {params.aperturesize} pixels")
    print(f"Off-axis center: {params.offaxis_center}")
    
    # ========== Step 2: 時系列ホログラムの読み込み ==========
    folder_path_blank = "/Volumes/QPI_0_.01_r/251126_kk/ph_6/Pos1/"
    N_FRAMES = 800
    
    print(f"\n=== Loading {N_FRAMES} holograms ===")
    holograms_blank = load_hologram_sequence(
        folder_path_blank,
        n_frames=N_FRAMES,
        crop_region=CROP_REGION
    )
    print(f"Loaded shape: {holograms_blank.shape}")
    
    # ========== Step 3: ALG感度の計算 ==========
    print("\n=== Calculating ALG sensitivity ===")
    sigma_alg_blank = calculate_ALG_sensitivity_shot_noise(
        hologram=holograms_blank[0],
        params=params,
        camera_gain=CAMERA_GAIN,
        filter_bandwidth_ratio=0.3
    )
    print(f"ALG calculated, mean: {np.mean(sigma_alg_blank):.6e} rad")
    
    # ========== Step 4: EXP感度の計算 ==========
    print("\n=== Calculating EXP sensitivity ===")
    sigma_exp_blank = calculate_EXP_sensitivity(
        holograms=holograms_blank,
        params=params,
        use_unwrap=True
    )
    print(f"EXP calculated, mean: {np.mean(sigma_exp_blank):.6e} rad")
    
    # ========== Step 5: Fig. 3(a-c)のプロット ==========
    print("\n=== Plotting Fig. 3 style ===")
    plot_fig3_style(
        sigma_exp=sigma_exp_blank,
        sigma_alg=sigma_alg_blank,
        wavelength=WAVELENGTH,
        vmax_factor=3.0,
        save_path="fig3_abc.png"
    )
    
    # ========== Step 6（オプション）: サンプルがある場合 ==========
    # path_sample = "/Volumes/QPI_0_.01_r/ph_21/Pos0/img_000000000_Default_000.tif"
    # img_sample = np.array(Image.open(path_sample))
    # img_sample = img_sample[CROP_REGION[0]:CROP_REGION[1], 
    #                         CROP_REGION[2]:CROP_REGION[3]]
    # 
    # # サンプルの位相再構成
    # field_sample = get_field(img_sample, params)
    # phase_sample = unwrap_phase(np.angle(field_sample))
    # intensity_sample = np.abs(field_sample)
    # 
    # # サンプルのALG感度
    # sigma_alg_sample = calculate_ALG_sensitivity_shot_noise(
    #     hologram=img_sample,
    #     params=params,
    #     camera_gain=CAMERA_GAIN,
    #     filter_bandwidth_ratio=0.3
    # )
    # 
    # # 完全版のFig. 3をプロット
    # plot_fig3_with_sample(
    #     sigma_exp_blank=sigma_exp_blank,
    #     sigma_alg_blank=sigma_alg_blank,
    #     sigma_alg_sample=sigma_alg_sample,
    #     intensity_sample=intensity_sample,
    #     phase_sample=phase_sample,
    #     wavelength=WAVELENGTH,
    #     save_path="fig3_complete.png"
    # )