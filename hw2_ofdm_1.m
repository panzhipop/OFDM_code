%% 參數設定
Nfft = 256;                  % FFT點數
CP = 16;                     % 循環字首長度
numSymbols = 20000;          % OFDM symbol 數量
SNRdB = 0:5:25;              % SNR 測試範圍
theta_true = 7;          % 延遲（取樣點, 問題二用）
epsilon_true = 0.2;      % 頻率偏移, 問題二用

%% 題目1:
fprintf('\n=== 題目1 ===\n');
ber_ideal = simulate_OFDM(Nfft, CP, numSymbols, SNRdB, 0, 0, 0);

%% 題目2:
fprintf('\n=== 題目2: 延遲(%d)和頻偏(%.2f) ===\n', theta_true, epsilon_true);
ber_offset = simulate_OFDM(Nfft, CP, numSymbols, SNRdB, theta_true, epsilon_true, 0);

%% 題目3:
fprintf('\n=== 題目3: 同步補償 ===\n');
ber_sync = simulate_OFDM(Nfft, CP, numSymbols, SNRdB, theta_true, epsilon_true, 1);

%% 畫出BER vs SNR
% 先轉 RGB 到 0~1
blue  = [48, 106, 199]/255;   % 藍色
yellow = [250, 213, 5]/255;   % 黃色
red   = [230, 87, 120]/255;   % 紅色

figure;
semilogy(SNRdB, ber_ideal, 'o-', 'Color', blue, 'LineWidth', 2);    % 藍
hold on;
semilogy(SNRdB, ber_offset, '-', 'Color', yellow, 'LineWidth', 2);   % 黃
semilogy(SNRdB, ber_sync, '--', 'Color', red, 'LineWidth', 2);       % 紅
grid on;
xlabel('SNR (dB)');
ylabel('Bit Error Rate (BER)');
legend('Ideal System', sprintf('With delay \\theta = %d and frequency offset \\epsilon = %.2f', theta_true, epsilon_true), ...
       'After synchronization compensation', 'Location', 'southwest', 'FontSize', 11);
title('OFDM 16QAM BER vs SNR');

%%
function ber = simulate_OFDM(Nfft, CP, numSymbols, SNRdB, theta, epsilon, S)
    
    M = 16;         % QAM調變階數 (16-QAM)
    bps = log2(M);  % Bits per symbol for QAM modulationbps = log2(M);

    % 產生隨機位元
    bits = randi([0 1], Nfft*bps, 1);
    
    % BER 儲存
    ber = zeros(size(SNRdB));
    
    % 計算第 idx BER點
    for idx = 1:length(SNRdB)
        SNR = SNRdB(idx);
        num_errors = 0;
        num_bits = 0;

        for i = 1:numSymbols
       
            % QAM 調變
            qamSymbols = qammod(bits, M, 'InputType', 'bit','UnitAveragePower',true);
        
            % OFDM 調變 (IFFT+加入CP)
            ifftData = ifft(qamSymbols, Nfft);
            tx = [ifftData(end-CP+1:end); ifftData];
            
            % 通道效應: 延遲和頻偏
            tx = apply_channel_effects(tx, theta, epsilon, Nfft);
    
            % AWGN 通道
            rx = awgn(tx, SNRdB(idx), 'measured');

            if S == 1 
                [theta_est, epsilon_est] = ML_synchronization(rx, Nfft, CP, SNR);
                % 根據估測的延遲找到 OFDM 符號的起始位置
                % theta_est 是 CP 開始的位置
                theta_est = round(theta_est);
                if theta_est < 0
                    theta_est = 0;
                end
                
                % 從 CP 開始位置提取完整的符號（包含 CP）
                symbol_start = theta_est + 1;
                symbol_end = symbol_start + CP + Nfft - 1;
                
                if symbol_end <= length(rx)
                    rx_symbols = rx(symbol_start:symbol_end);
                else
                    % 如果超出範圍，使用預設位置
                    rx_symbol = rx(1:min(CP+Nfft, length(rx)));
                    if length(rx_symbol) < CP+Nfft
                        rx_symbols = [rx_symbol; zeros(CP+Nfft-length(rx_symbol), 1)];
                    end
                end
                
                % 9. 補償頻偏（對整個符號，包含 CP）
                k = 0:length(rx_symbols)-1;
                freq_compensation = exp(-1j*2*pi*epsilon_est*k(:)/Nfft);
                rx = rx_symbols .* freq_compensation;
            end
        
            % OFDM 解調 (去CP+FFT)
            rxNoCP = rx(CP+1:end);
            rxSymbols = fft(rxNoCP, Nfft);
         
            % QAM 解調
            rxBits = qamdemod(rxSymbols, M, 'OutputType', 'bit','UnitAveragePower',true);
            num_errors = num_errors + sum(bits ~= rxBits);
            num_bits = num_bits + (Nfft * bps);
         
         end
         % 計算BER
         ber(idx) = num_errors / num_bits;
         fprintf('SNR = %2d dB -> BER = %.5e\n', SNRdB(idx), ber(idx));
    end
end

%%
function tx = apply_channel_effects(tx_signal, theta, epsilon, Nfft)
    % 應用通道效應：延遲和頻偏
    
    % 1. 先應用頻偏（對原始信號）
    k = 0:length(tx_signal)-1;
    freq_offset = exp(1j*2*pi*epsilon*k(:)/Nfft);
    tx = tx_signal .* freq_offset;
    
    % 2. 再應用延遲
    if theta > 0
        tx = [zeros(theta, 1); tx];
    end
end

%%
function [theta_ML, epsilon_ML, Lambda, phi_vals] = ML_synchronization(r, Nfft, L, SNRdB)
% ML_synchronization  最大似然時間/頻率偏移估測 (van de Beek et al.)
%
% Inputs:
%   r      - 接收到的時域樣本向量（複數），假設包含至少一個完整 OFDM 符號 (包括 CP)
%   Nfft   - OFDM IFFT/FFT 點數 (論文的 N)
%   L      - 循環前綴長度 CP (論文的 L)
%   SNRdB  - (可選) SNR in dB，用於計算 rho；若不提供則假設高 SNR (rho->1)
%
% Outputs:
%   theta_ML   - 估測出的時間偏移 (0-based index, 對應 CP 的起始位置)
%   epsilon_ML - 估測出的頻偏 (normalized frequency offset, fraction of subcarrier spacing)
%   Lambda     - 搜索窗口中每個 theta 的目標函數值 (列向量)
%   phi_vals   - 對應每個 theta 的 complex phi(\theta) 值 (列向量)
%
% Reference: J.-J. van de Beek, M. Sandell, P.O. Börjesson, "ML Estimation of Time
% and Frequency Offset in OFDM Systems", IEEE Trans. Signal Processing, 1997.
%
% Example:
%   [theta, eps] = ML_synchronization(r, 256, 16, 10);
%
% Note: 函式假定 r 的時間索引以 0-based 想像，輸入向量在 MATLAB 中仍為 1-based。
%

% Input checks
if nargin < 3
    error('Usage: [theta,eps] = ML_synchronization(r, Nfft, L, SNRdB)');
end
if nargin < 4
    SNRdB = []; % allow omitted SNR
end

N = Nfft;

% compute rho from SNR if provided: rho = SNR/(1+SNR)
if ~isempty(SNRdB)
    SNR_linear = 10^(SNRdB/10);
    rho = SNR_linear / (1 + SNR_linear);
else
    % if SNR not given, set rho to a reasonable default (e.g., 0.9)
    % or you can set rho = 0 to ignore the energy term (not recommended)
    rho = 0.9;
end

% Determine valid theta range:
% need indices i and i+N to be inside r, for i = theta .. theta+L-1
% so require theta + L - 1 + N <= length(r) - 1 (0-based)
% convert to MATLAB 1-based indexing required later; compute max theta (0-based)
lenr = length(r);
theta_max = lenr - N - L;    % 0-based maximum theta
if theta_max < 0
    error('Received vector r too short for given Nfft and L.');
end

theta_cands = 0:theta_max;
numCands = length(theta_cands);

% preallocate
phi_vals = complex(zeros(numCands,1));
Lambda = zeros(numCands,1);

% For each candidate theta, compute phi and energy term
for idx = 1:numCands
    theta = theta_cands(idx);          % 0-based
    phi_sum = 0 + 0j;
    energy_sum = 0;
    % sum over i = theta .. theta+L-1  (0-based)
    for k = 0:(L-1)
        idx_cp = theta + k + 1;            % MATLAB index for r(theta+k)
        idx_data = theta + k + N + 1;      % MATLAB index for r(theta+k+N)
        % safety check (should be inside by construction)
        if idx_cp <= lenr && idx_data <= lenr
            % phi_sum = phi_sum + conj(r(idx_cp)) * r(idx_data);
            phi_sum = phi_sum + conj(r(idx_data)) * r(idx_cp);
            energy_sum = energy_sum + (abs(r(idx_cp))^2 + abs(r(idx_data))^2);
        end
    end
    phi_vals(idx) = phi_sum;
    Phi = energy_sum / 2;    % as in paper: 0.5 * sum(|r(i)|^2 + |r(i+N)|^2)
    % compressed log-likelihood (up to additive constants)
    Lambda(idx) = abs(phi_sum) - rho * Phi;
end

% pick theta that maximizes Lambda
[~, max_idx] = max(Lambda);
theta_ML = theta_cands(max_idx);   % 0-based return (consistent with paper notation)

% frequency offset estimate from argument of phi at theta_ML
phi_at = phi_vals(max_idx);
if abs(phi_at) > 0
    epsilon_ML = - angle(phi_at) / (2*pi);  % normalized (fraction of subcarrier spacing)
    % wrap epsilon to [-0.5, 0.5)
    epsilon_ML = mod(epsilon_ML + 0.5, 1) - 0.5;
else
    epsilon_ML = 0;
end

end