# %%
"""
Temporal Noise評価の動作デモ
実際のデータなしでシミュレーションで理解する
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# %% ========== 1. ホログラムの生成（シミュレーション） ==========

def generate_hologram(M, N, k_M, k_N, alpha, beta, phi_s, camera_gain, add_noise=True):
    """
    論文 Eq. (1) に基づくホログラム生成
    
    I(m,n) = α + β·cos(φ_s - k_M·m - k_N·n) + noise
    
    Args:
        M, N: ホログラムサイズ
        k_M, k_N: キャリア周波数（正規化）
        alpha: DC成分（2D配列）
        beta: 干渉縞振幅（2D配列）
        phi_s: サンプル位相（2D配列）
        camera_gain: カメラゲイン [e-/ADU]
        add_noise: Poissonノイズを加えるか
    """
    m, n = np.meshgrid(np.arange(N), np.arange(M))
    
    # キャリア位相
    carrier_phase = 2 * np.pi * (k_M * m + k_N * n) / M
    
    # 理想的なホログラム
    I_ideal = alpha + 2 * beta * np.cos(phi_s - carrier_phase)
    
    if add_noise:
        # Poissonノイズ追加
        # 光子数 = I * g → Poisson分布 → ADU = photons / g
        I_noisy = np.random.poisson(I_ideal * camera_gain) / camera_gain
        return I_noisy
    else:
        return I_ideal


# パラメータ設定
M = N = 512
k_M = k_N = 128  # キャリア周波数
camera_gain = 30.0  # e-/ADU

# Gaussian照明パターン
y, x = np.ogrid[-M//2:M//2, -N//2:N//2]
sigma_illum = M / 4
alpha = 800 * np.exp(-(x**2 + y**2) / (2 * sigma_illum**2))
beta = 0.5 * alpha  # 視認性 V = 2β/α = 1.0

# サンプル位相（中央に正方形の位相物体）
phi_s = np.zeros((M, N))
phi_s[M//3:2*M//3, N//3:2*N//3] = 1.0  # 1 radian

# ホログラム生成
hologram = generate_hologram(M, N, k_M, k_N, alpha, beta, phi_s, camera_gain)

# 表示
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

im0 = axes[0].imshow(alpha, cmap='gray')
axes[0].set_title('DC intensity (α)')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(phi_s, cmap='twilight')
axes[1].set_title('Sample phase (φ_s)')
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(hologram, cmap='gray')
axes[2].set_title('Off-axis Hologram (with noise)')
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()

print(f"ホログラム統計:")
print(f"  Mean intensity: {np.mean(hologram):.1f} ADU")
print(f"  Std intensity: {np.std(hologram):.1f} ADU")
print(f"  Expected shot noise std: {np.sqrt(np.mean(hologram)/camera_gain):.1f} ADU")


# %% ========== 2. FFTでスペクトルを確認 ==========

def visualize_fft_spectrum(hologram, k_M, k_N, M, N):
    """ホログラムのFFTスペクトルを可視化"""
    
    # FFT
    fft = np.fft.fft2(hologram)
    fft_shifted = np.fft.fftshift(fft)
    spectrum = np.abs(fft_shifted)
    
    # 対数スケールで表示
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(np.log(spectrum + 1), cmap='hot', extent=[-N//2, N//2, -M//2, M//2])
    
    # DC成分に円を描画
    circle_dc = Circle((0, 0), 30, fill=False, edgecolor='cyan', linewidth=2, label='DC (0th order)')
    ax.add_patch(circle_dc)
    
    # サイドバンドに円を描画
    circle_sb = Circle((k_N, k_M), 30, fill=False, edgecolor='lime', linewidth=2, label='Sideband (1st order)')
    ax.add_patch(circle_sb)
    
    ax.set_title('FFT Spectrum of Hologram', fontsize=14, fontweight='bold')
    ax.set_xlabel('k_N (frequency in x)')
    ax.set_ylabel('k_M (frequency in y)')
    ax.legend(loc='upper right')
    plt.colorbar(im, ax=ax, label='log(|FFT|)')
    plt.show()
    
    print(f"\nスペクトルのピーク位置:")
    print(f"  DC (中心): (0, 0)")
    print(f"  Sideband: ({k_N}, {k_M})")
    print(f"  キャリア周波数までの距離: {np.sqrt(k_M**2 + k_N**2):.1f} pixels")

visualize_fft_spectrum(hologram, k_M, k_N, M, N)


# %% ========== 3. α と β の抽出（論文の重要な部分） ==========

def extract_alpha_beta_demo(hologram, M, N, k_M, k_N, aperture_radius=30):
    """
    DC成分（α）とサイドバンド振幅（β）の抽出を可視化
    """
    # FFT
    fft = np.fft.fft2(hologram)
    fft_shifted = np.fft.fftshift(fft)
    
    # === DC成分（α）の抽出 ===
    # 中心にディスク型マスク
    yy, xx = np.ogrid[-M//2:M//2, -N//2:N//2]
    dc_mask = (xx**2 + yy**2) < aperture_radius**2
    
    dc_fft = fft_shifted * dc_mask
    dc_field = np.fft.ifft2(np.fft.ifftshift(dc_fft))
    
    # スケーリング補正
    scale = 2 * aperture_radius / M
    alpha_extracted = np.abs(dc_field) * scale**2
    
    # === サイドバンド（β）の抽出 ===
    # off-axis位置にディスク型マスク
    sb_mask = ((xx - k_N)**2 + (yy - k_M)**2) < aperture_radius**2
    
    sb_fft = fft_shifted * sb_mask
    sb_field = np.fft.ifft2(np.fft.ifftshift(sb_fft))
    beta_extracted = np.abs(sb_field) * scale**2
    
    # 可視化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 上段：FFTとマスク
    axes[0, 0].imshow(np.log(np.abs(fft_shifted) + 1), cmap='hot')
    axes[0, 0].set_title('Original FFT spectrum')
    
    axes[0, 1].imshow(dc_mask, cmap='gray')
    axes[0, 1].set_title('DC mask (0th order)')
    circle = Circle((N//2, M//2), aperture_radius, fill=False, color='cyan', linewidth=2)
    axes[0, 1].add_patch(circle)
    
    axes[0, 2].imshow(sb_mask, cmap='gray')
    axes[0, 2].set_title('Sideband mask (1st order)')
    circle = Circle((N//2 + k_N, M//2 + k_M), aperture_radius, fill=False, color='lime', linewidth=2)
    axes[0, 2].add_patch(circle)
    
    # 下段：抽出結果
    im1 = axes[1, 0].imshow(alpha, cmap='gray')
    axes[1, 0].set_title('Ground truth α')
    plt.colorbar(im1, ax=axes[1, 0])
    
    im2 = axes[1, 1].imshow(alpha_extracted, cmap='gray')
    axes[1, 1].set_title('Extracted α')
    plt.colorbar(im2, ax=axes[1, 1])
    
    im3 = axes[1, 2].imshow(beta_extracted, cmap='gray')
    axes[1, 2].set_title('Extracted β')
    plt.colorbar(im3, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.show()
    
    # 精度確認
    print(f"\nα抽出精度:")
    print(f"  Ground truth mean: {np.mean(alpha):.2f}")
    print(f"  Extracted mean: {np.mean(alpha_extracted):.2f}")
    print(f"  Relative error: {np.abs(np.mean(alpha - alpha_extracted))/np.mean(alpha)*100:.2f}%")
    
    print(f"\nβ抽出精度:")
    print(f"  Ground truth mean: {np.mean(beta):.2f}")
    print(f"  Extracted mean: {np.mean(beta_extracted):.2f}")
    print(f"  Relative error: {np.abs(np.mean(beta - beta_extracted))/np.mean(beta)*100:.2f}%")
    
    return alpha_extracted, beta_extracted

alpha_ext, beta_ext = extract_alpha_beta_demo(hologram, M, N, k_M, k_N)


# %% ========== 4. ALG感度の計算（論文 Eq. 12） ==========

def calculate_ALG_demo(alpha, beta, M, N, camera_gain, filter_bandwidth_ratio=0.3):
    """
    論文 Eq. (12) の実装と可視化
    
    σ_φ = sqrt(S * α / (2 * g * M * N * β²))
    """
    # キャリア周波数までの距離
    carrier_distance = np.sqrt(k_M**2 + k_N**2)
    
    # フィルタ半径とフィルタ開口面積
    radius = filter_bandwidth_ratio * carrier_distance
    S = np.pi * radius**2
    
    print(f"\n=== ALG計算パラメータ ===")
    print(f"キャリア周波数距離: {carrier_distance:.1f} pixels")
    print(f"フィルタ半径: {radius:.1f} pixels")
    print(f"フィルタ開口面積 S: {S:.0f} pixels")
    print(f"ホログラムサイズ M×N: {M}×{N} = {M*N} pixels")
    print(f"S/(M×N): {S/(M*N)*100:.2f}%")
    print(f"カメラゲイン g: {camera_gain} e-/ADU")
    
    # Eq. (12) の計算
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_phi = np.sqrt(S * alpha / (2 * camera_gain * M * N * beta**2))
        sigma_phi[~np.isfinite(sigma_phi)] = 0
    
    # 統計
    mean_sigma = np.mean(sigma_phi[sigma_phi > 0])
    
    print(f"\n=== ALG感度結果 ===")
    print(f"Mean σ_φ: {mean_sigma:.6e} rad = {mean_sigma*1000:.3f} mrad")
    print(f"Min σ_φ: {np.min(sigma_phi[sigma_phi > 0]):.6e} rad")
    print(f"Max σ_φ: {np.max(sigma_phi):.6e} rad")
    
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    im0 = axes[0].imshow(alpha, cmap='gray')
    axes[0].set_title('α (DC intensity)')
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(beta, cmap='gray')
    axes[1].set_title('β (fringe amplitude)')
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(sigma_phi, cmap='hot')
    axes[2].set_title(f'ALG sensitivity σ_φ (mean={mean_sigma*1000:.3f} mrad)')
    plt.colorbar(im2, ax=axes[2], label='σ_φ (rad)')
    
    plt.tight_layout()
    plt.show()
    
    return sigma_phi

sigma_alg = calculate_ALG_demo(alpha_ext, beta_ext, M, N, camera_gain)


# %% ========== 5. EXP感度の計算（時系列シミュレーション） ==========

def calculate_EXP_demo(M, N, k_M, k_N, alpha, beta, phi_s, camera_gain, n_frames=100):
    """
    時系列ホログラムからEXP感度を計算
    """
    print(f"\n=== {n_frames}フレームのホログラムを生成中... ===")
    
    phases = []
    
    for i in range(n_frames):
        if i % 20 == 0:
            print(f"  Frame {i}/{n_frames}")
        
        # ノイズ入りホログラム生成
        holo = generate_hologram(M, N, k_M, k_N, alpha, beta, phi_s, camera_gain, add_noise=True)
        
        # 位相再構成（簡易版）
        fft = np.fft.fft2(holo)
        fft_shifted = np.fft.fftshift(fft)
        
        # サイドバンドフィルタリング
        yy, xx = np.ogrid[-M//2:M//2, -N//2:N//2]
        sb_mask = ((xx - k_N)**2 + (yy - k_M)**2) < 30**2
        
        sb_fft = fft_shifted * sb_mask
        sb_field = np.fft.ifft2(np.fft.ifftshift(sb_fft))
        
        phase = np.angle(sb_field)
        phases.append(phase)
    
    phases = np.array(phases)
    
    # 時間方向の標準偏差
    sigma_exp = np.std(phases, axis=0)
    
    mean_sigma = np.mean(sigma_exp[sigma_exp > 0])
    
    print(f"\n=== EXP感度結果 ===")
    print(f"Mean σ_φ: {mean_sigma:.6e} rad = {mean_sigma*1000:.3f} mrad")
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im0 = axes[0].imshow(sigma_exp, cmap='hot')
    axes[0].set_title(f'EXP sensitivity (mean={mean_sigma*1000:.3f} mrad)')
    plt.colorbar(im0, ax=axes[0], label='σ_φ (rad)')
    
    # 時間変動の例（1ピクセル）
    pixel_phases = phases[:, M//2, N//2]
    axes[1].plot(pixel_phases, 'b-', linewidth=0.5, alpha=0.7)
    axes[1].axhline(np.mean(pixel_phases), color='r', linestyle='--', label='Mean')
    axes[1].fill_between(range(n_frames), 
                         np.mean(pixel_phases) - sigma_exp[M//2, N//2],
                         np.mean(pixel_phases) + sigma_exp[M//2, N//2],
                         alpha=0.3, color='red', label='±1σ')
    axes[1].set_xlabel('Frame number')
    axes[1].set_ylabel('Phase (rad)')
    axes[1].set_title(f'Temporal phase variation at center pixel')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return sigma_exp

sigma_exp = calculate_EXP_demo(M, N, k_M, k_N, alpha, beta, phi_s, camera_gain, n_frames=200)


# %% ========== 6. システム効率の評価 ==========

def evaluate_system_efficiency(sigma_exp, sigma_alg):
    """
    システム効率 = ALG / EXP × 100%
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        efficiency = sigma_alg / sigma_exp * 100
        efficiency[~np.isfinite(efficiency)] = 0
        efficiency = np.clip(efficiency, 0, 100)
    
    mean_eff = np.mean(efficiency[efficiency > 0])
    
    print(f"\n=== システム効率 ===")
    print(f"Mean efficiency: {mean_eff:.2f}%")
    print(f"Min efficiency: {np.min(efficiency[efficiency > 0]):.2f}%")
    print(f"Max efficiency: {np.max(efficiency):.2f}%")
    
    if mean_eff > 95:
        print("→ 優秀！システムはショットノイズ限界に近い")
    elif mean_eff > 90:
        print("→ 良好。わずかな改善余地あり")
    else:
        print("→ 要改善。システムノイズが支配的")
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    im0 = axes[0, 0].imshow(sigma_exp, cmap='hot')
    axes[0, 0].set_title('EXP sensitivity')
    plt.colorbar(im0, ax=axes[0, 0], label='σ_φ (rad)')
    
    im1 = axes[0, 1].imshow(sigma_alg, cmap='hot')
    axes[0, 1].set_title('ALG sensitivity')
    plt.colorbar(im1, ax=axes[0, 1], label='σ_φ (rad)')
    
    # 中央列の比較
    center_col = M // 2
    axes[1, 0].plot(sigma_exp[:, center_col], 'b-', label='EXP', linewidth=2)
    axes[1, 0].plot(sigma_alg[:, center_col], 'r--', label='ALG', linewidth=2)
    axes[1, 0].set_xlabel('Pixel Y')
    axes[1, 0].set_ylabel('σ_φ (rad)')
    axes[1, 0].set_title('Center column comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 効率マップ
    im3 = axes[1, 1].imshow(efficiency, cmap='RdYlGn', vmin=80, vmax=100)
    axes[1, 1].set_title(f'System Efficiency (mean={mean_eff:.1f}%)')
    plt.colorbar(im3, ax=axes[1, 1], label='Efficiency (%)')
    
    plt.tight_layout()
    plt.show()

evaluate_system_efficiency(sigma_exp, sigma_alg)

print("\n" + "="*60)
print("デモ完了！")
print("="*60)
# %%
