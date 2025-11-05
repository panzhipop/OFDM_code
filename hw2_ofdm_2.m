%% ===== Fig.6 模擬: ML estimator MSE (time & freq) =====
clear; clc;
Nfft = 256;
SNRdB = 0:5:25;
numTrials = 100000;     % Monte Carlo 次數，可調 100~1000 取決於效能
L_values = [4 8 15]; % 不同 CP 長度
epsilon_true = 0.2;  % 頻偏
theta_true = 20;      % 時間偏移

mse_time = zeros(length(L_values), length(SNRdB));
mse_freq = zeros(length(L_values), length(SNRdB));

fprintf('=== 模擬 Fig.6 (AWGN channel) ===\n');
for li = 1:length(L_values)
    L = L_values(li);
    fprintf('\n--- L = %d ---\n', L);
    for si = 1:length(SNRdB)
        SNR = SNRdB(si);
        theta_err = zeros(numTrials,1);
        eps_err = zeros(numTrials,1);

        for trial = 1:numTrials
            % ===== 產生一個 OFDM 符號 =====
            x = randn(Nfft,1) + 1j*randn(Nfft,1);
            x = x / sqrt(mean(abs(x).^2)); % normalize power

            tx = [x(end-L+1:end); x]; % 加入循環字首 CP
            
            % ===== 通道效應：延遲 + 頻偏 =====
            k = 0:length(tx)-1;
            freq_offset = exp(1j*2*pi*epsilon_true*k(:)/Nfft);
            tx = tx .* freq_offset;   % 加入頻偏
            tx = [zeros(theta_true,1); tx]; % 加入時間延遲

            % ===== AWGN channel =====
            rx = awgn(tx, SNR, 'measured');

            % ===== ML estimator =====
            [theta_ML, epsilon_ML] = ML_synchronization(rx, Nfft, L, SNR);
            
            % 儲存平方誤差
            theta_err(trial) = (theta_ML - theta_true)^2;
            eps_err(trial) = (epsilon_ML - epsilon_true)^2;
        end

        mse_time(li, si) = mean(theta_err);
        mse_freq(li, si) = mean(eps_err);
        fprintf('SNR=%2d dB => MSE_t=%.3g, MSE_f=%.3g\n', ...
            SNR, mse_time(li,si), mse_freq(li,si));
    end
end

%% ===== 繪圖: Time offset estimator MSE =====
figure;
for li = 1:length(L_values)
    semilogy(SNRdB, mse_time(li,:), 'LineWidth', 2); hold on;
end
grid on;
xlabel('SNR (dB)');
ylabel('MSE of time estimator');
legend('L=4','L=8','L=15','Location','southwest');
title('Fig.6(a) Time estimator MSE (AWGN channel)');

%% ===== 繪圖: Frequency offset estimator MSE =====
figure;
for li = 1:length(L_values)
    semilogy(SNRdB, mse_freq(li,:), 'LineWidth', 2); hold on;
end
grid on;
xlabel('SNR (dB)');
ylabel('MSE of frequency estimator');
legend('L=4','L=8','L=15','Location','northeast');
title('Fig.6(b) Frequency estimator MSE (AWGN channel)');


%% ===== ML_synchronization 函式 =====
function [theta_ML, epsilon_ML, Lambda, phi_vals] = ML_synchronization(r, Nfft, L, SNRdB)
% ML_synchronization  最大似然時間/頻率偏移估測 (van de Beek et al., 1997)

N = Nfft;

% compute rho from SNR if provided: rho = SNR/(1+SNR)
if nargin >= 4 && ~isempty(SNRdB)
    SNR_linear = 10^(SNRdB/10);
    rho = SNR_linear / (1 + SNR_linear);
else
    rho = 0.9;
end

lenr = length(r);
theta_max = lenr - N - L;    % 0-based maximum theta
if theta_max < 0
    error('Received vector r too short for given Nfft and L.');
end

theta_cands = 0:theta_max;
numCands = length(theta_cands);

phi_vals = complex(zeros(numCands,1));
Lambda = zeros(numCands,1);

% ML 搜索
for idx = 1:numCands
    theta = theta_cands(idx);
    phi_sum = 0;
    energy_sum = 0;
    for k = 0:(L-1)
        idx_cp = theta + k + 1;
        idx_data = theta + k + N + 1;
        if idx_data <= lenr
            phi_sum = phi_sum + conj(r(idx_data)) * r(idx_cp);
            energy_sum = energy_sum + (abs(r(idx_cp))^2 + abs(r(idx_data))^2);
        end
    end
    phi_vals(idx) = phi_sum;
    Phi = energy_sum / 2;
    Lambda(idx) = abs(phi_sum) - rho * Phi;
end

% 最大化 Λ 取 theta_ML
[~, max_idx] = max(Lambda);
theta_ML = theta_cands(max_idx);

% 頻率偏移估測
phi_at = phi_vals(max_idx);
if abs(phi_at) > 0
    epsilon_ML = - angle(phi_at) / (2*pi);
    epsilon_ML = mod(epsilon_ML + 0.5, 1) - 0.5;  % wrap to [-0.5, 0.5)
else
    epsilon_ML = 0;
end
end
