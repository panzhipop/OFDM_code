%% =========================
%    OFDM 主程式 
% =========================

clear; clc;

%% 參數設定
Nfft = 256;                  % FFT點數
CP = 16;                     % 循環字首長度
numSymbols = 20000;          % OFDM symbol 數量
SNRdB = 0:5:25;              % SNR 測試範圍
theta_true = 7;              % 延遲（取樣點）
epsilon_true = 0.2;          % 頻率偏移

M = 16;                      % 16QAM
bps = log2(M);

%% 三個實驗共用同一組 bits
bits = randi([0 1], Nfft*bps, 1);

% 畫出 bits 的長條圖
figure;
bar(0:length(bits)-1, bits);        % bar chart
xlabel('Index');
ylabel('Bit Value');
ylim([-0.5 1.5]);                   % 讓 0/1 更清楚
title('Bit Sequence Bar Chart');

fprintf('\n=== 題目1: Ideal ===\n');
ber_ideal = simulate_OFDM(Nfft, CP, numSymbols, SNRdB, 0, 0, 0, bits);

fprintf('\n=== 題目2: 有延遲 %d & 頻偏 %.2f ===\n', theta_true, epsilon_true);
ber_offset = simulate_OFDM(Nfft, CP, numSymbols, SNRdB, theta_true, epsilon_true, 0, bits);

fprintf('\n=== 題目3: 同步補償 ===\n');
ber_sync = simulate_OFDM(Nfft, CP, numSymbols, SNRdB, theta_true, epsilon_true, 1, bits);

%% 畫 BER 圖
blue  = [48, 106, 199]/255;
yellow = [250, 213, 5]/255;
red   = [230, 87, 120]/255;

figure;
semilogy(SNRdB, ber_ideal, 'o-', 'Color', blue, 'LineWidth', 2);
hold on;
semilogy(SNRdB, ber_offset, '-', 'Color', yellow, 'LineWidth', 2);
semilogy(SNRdB, ber_sync, '--', 'Color', red, 'LineWidth', 2);
grid on;

xlabel('SNR (dB)');
ylabel('Bit Error Rate (BER)');
legend('Ideal System', ...
    sprintf('With delay \\theta = %d and frequency offset \\epsilon = %.2f', theta_true, epsilon_true), ...
    'After synchronization compensation', ...
    'Location', 'southwest', 'FontSize', 11);

title('OFDM BER vs SNR');


%% =========================
%       OFDM 模擬函式
% =========================
function ber = simulate_OFDM(Nfft, CP, numSymbols, SNRdB, theta, epsilon, S, bits)
    
    M = 16;
    bps = log2(M);

    % ✅ bits 改為由外部傳入，不在此生成

    ber = zeros(size(SNRdB));
    
    for idx = 1:length(SNRdB)
        SNR = SNRdB(idx);
        num_errors = 0;
        num_bits = 0;

        for i = 1:numSymbols
       
            % QAM 調變
            qamSymbols = qammod(bits, M, 'InputType', 'bit','UnitAveragePower',true);
        
            % OFDM 調變
            ifftData = ifft(qamSymbols, Nfft);
            tx = [ifftData(end-CP+1:end); ifftData];
            
            % 加延遲 + 頻偏
            tx = apply_channel_effects(tx, theta, epsilon, Nfft);
    
            % AWGN
            rx = awgn(tx, SNRdB(idx), 'measured');

            %% === 若要做同步補償 ===
            if S == 1 
                [theta_est, epsilon_est] = ML_synchronization(rx, Nfft, CP, SNR);

                theta_est = round(theta_est);
                if theta_est < 0, theta_est = 0; end

                % 取出 OFDM 符號（含 CP）
                symbol_start = theta_est + 1;
                symbol_end   = symbol_start + CP + Nfft - 1;

                if symbol_end <= length(rx)
                    rx_symbols = rx(symbol_start:symbol_end);
                else
                    rx_symbol = rx(1:min(CP+Nfft, length(rx)));
                    if length(rx_symbol) < CP+Nfft
                        rx_symbols = [rx_symbol; zeros(CP+Nfft-length(rx_symbol), 1)];
                    end
                end
                
                % 補償頻偏
                k = 0:length(rx_symbols)-1;
                freq_compensation = exp(-1j * 2*pi * epsilon_est * k(:) / Nfft);
                rx = rx_symbols .* freq_compensation;
            end
        
            %% OFDM 解調
            rxNoCP = rx(CP+1:end);
            rxSymbols = fft(rxNoCP, Nfft);
         
            % QAM 解調
            rxBits = qamdemod(rxSymbols, M, 'OutputType', 'bit','UnitAveragePower',true);

            % 計算 BER
            num_errors = num_errors + sum(bits ~= rxBits);
            num_bits   = num_bits   + (Nfft * bps);
         
        end

        ber(idx) = num_errors / num_bits;
        fprintf('SNR = %2d dB -> BER = %.5e\n', SNRdB(idx), ber(idx));
    end
end


%% =========================
%      加延遲 + 頻偏
% =========================
function tx = apply_channel_effects(tx_signal, theta, epsilon, Nfft)

    % 頻偏
    k = 0:length(tx_signal)-1;
    freq_offset = exp(1j*2*pi*epsilon*k(:)/Nfft);
    tx = tx_signal .* freq_offset;
    
    % 加延遲（zero-padding）
    if theta > 0
        tx = [zeros(theta, 1); tx];
    end
end


%% =========================
%    ML 同步估測 (van de Beek)
% =========================
function [theta_ML, epsilon_ML, Lambda, phi_vals] = ML_synchronization(r, Nfft, L, SNRdB)

N = Nfft;
lenr = length(r);

SNR_linear = 10^(SNRdB/10);
rho = SNR_linear / (1 + SNR_linear);

% 基本上是會有 theta 個 ， 如題目條件 eg. theta = 7
theta_max = lenr - N - L;
theta_cands = 0:theta_max;
numCands = length(theta_cands);

phi_vals = zeros(numCands,1);
Lambda = zeros(numCands,1);

for idx = 1:numCands
    theta = theta_cands(idx);

    phi_sum = 0;
    energy_sum = 0;

    % 這邊在做公式 (6) 、 (7)
    for k = 0:(L-1)
        i1 = theta + k + 1;
        i2 = theta + k + N + 1;

        % correlation part
        phi_sum = phi_sum + conj(r(i2)) * r(i1);

        % energy part
        energy_sum = energy_sum + (abs(r(i1))^2 + abs(r(i2))^2);
    end

    phi_vals(idx) = phi_sum;
    Phi = energy_sum / 2;
    Lambda(idx) = abs(phi_sum) - rho * Phi;
end

% 找到最大值的idx
[~, max_idx] = max(Lambda);
theta_ML = theta_cands(max_idx);

% 這邊在做公式 (12) 、 (13)
% phi 和 epsilon 的 Maxlikelihood estimate
phi_at = phi_vals(max_idx);
epsilon_ML = -angle(phi_at)/(2*pi);
epsilon_ML = mod(epsilon_ML + 0.5, 1) - 0.5;

end
