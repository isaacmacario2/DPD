clear;
%% Parâmetros
bw = 100e6;                           % Hz
symPerFrame = 1;                      % Symbols per frame
M = 16;                               % 16-QAM 
osf = 5;                              % Oversampling factor
spacing = 30e3;                       % Espaçamento
numCarr = 3276;                       % Subportadoras
cycPrefLen = 288;                     % Prefixo cíclico
fftLength = 2^ceil(log2(bw/spacing)); % 4096
numGBCarr = fftLength - numCarr;      % Intervalo de guarda (820)
fs = spacing*fftLength*osf;           % Frequência de amostragem

numDataCarr = fftLength - numGBCarr - 1; % Data = 4096-820-1 = 3275
nullIdx = [1:numGBCarr/2+1 fftLength-numGBCarr/2+1:fftLength]';

%% Gerando sinal aleatótio
x = randi([0 M-1],numDataCarr,symPerFrame);

%% Modulação QAM
qamModOut = qammod(x,M,"UnitAveragePower",true);%*sqrt(pin*100);

%% OFDM
ofdmModOut = ofdmmod(qamModOut/osf,fftLength,cycPrefLen,nullIdx,OversamplingFactor=osf);

%% Ajuste de potência
pindBm = 0;
pin = 10.^((pindBm-30)/10); % dBm -> Watts

% ofdmModOut = ofdmModOut*sqrt(pin*50);%/norm(ofdmModOut);
% ofdmModOut = ofdmModOut*sqrt(pin*100);%/norm(ofdmModOut);

% RMS
ofdmModOut = ofdmModOut*sqrt(pin*100*length(ofdmModOut))/norm(ofdmModOut);

% Potência média do sinal OFDM
% current_power = mean(abs(ofdmModOut).^2);
% desired_power = pin; % Potência em watts
% scaling_factor = sqrt(desired_power / current_power);
% ofdmModOut = ofdmModOut * scaling_factor;

%% Espectro
% [rx,~]= xcorr(ofdmModOut,'normalized');
% [~, I] = max(abs(rx));
% % [ofdmModOut_freq,w] = freqz(ofdmModOut,1,10*length(ofdmModOut),'whole');
% rx_cut1 = rx(I-1000:I+1000).*hamming(2001);
% rx_cut2 = rx(I-100:I+100);
% rx_cut3 = rx(I-10:I+10);
% [Rx1,wr1] = freqz(rx_cut1,1,length(rx),'whole');
% [Rx2,wr2] = freqz(rx_cut2,1,length(rx),'whole');
% [Rx3,wr3] = freqz(rx_cut3,1,length(rx),'whole');
% figure
% subplot(3,1,1)
% plot((fs/2)*(wr1-pi)/(pi*1e6),20*log10(abs(fftshift(Rx1))))
% title('rx com 2001 amostras')
% xlabel('f (MHz)')
% ylabel('dB')
% axis([-300 300 -100 20])
% subplot(3,1,2)
% plot((fs/2)*(wr2-pi)/(pi*1e6),20*log10(abs(fftshift(Rx2))))
% title('rx com 201 amostras')
% xlabel('f (MHz)')
% ylabel('dB')
% axis([-300 300 -100 20])
% subplot(3,1,3)
% plot((fs/2)*(wr3-pi)/(pi*1e6),20*log10(abs(fftshift(Rx3))))
% title('rx com 21 amostras')
% xlabel('f (MHz)')
% ylabel('dB')
% axis([-300 300 -100 20])
 
% specAn = spectrumAnalyzer("NumInputPorts",1,"ShowLegend",true,ChannelNames={"Sinal gerado"},SampleRate=fs);
% specAn(ofdmModOut)

% % Definir o número de pontos da FFT
% N = length(ofdmModOut);
% 
% window = hamming(N);
% % window  = kaiser(N, 9);
% ofdmModOut_w = ofdmModOut .* window;
% 
% % Calcular a FFT do sinal
% X = fftshift(fft(ofdmModOut_w, N)); 
% X = fftshift(pwelch(ofdmModOut_w, N)); 
% 
% % Calcular a potência do sinal
% Pot = abs(X).^2/(N); %
% 
% % Converter para dBm
% P_signal_dBm = 10*log10(Pot) + 30; 
% 
% % Vetor de frequências
% f = (-fs/2:fs/N:fs/2-fs/N);
% 
% % Plotar o espectro
% figure;
% plot(f/1e6, P_signal_dBm, 'LineWidth', 1.5);
% xlabel('Frequência (MHz)');
% ylabel('Potência (dBm)');
% title('Espectro do Sinal em dBm');
% grid on;
% xlim([-fs/2 fs/2]/1e6); 

% espectro(1024,fs,ofdmModOut,'pwelch')
% return

%% rf.PAmemory
load('matrizCoef3x3CT.mat')
rfpa = rf.PAmemory(CoefficientMatrix=fitCoefMatMem);
% % rfpa = rf.PAmemory(CoefficientMatrix=M_coeffs_MP);
% % rfpa = rf.PAmemory (Model='Cross-term Memory', CoefficientMatrix=fitCoefMat);
out_memory = rfpa(ofdmModOut);
% out_memory = rfpa(results.InputWaveform);

%% PA sem memória
% gain = 10;
% 
% % PA sem memória
% pa = comm.MemorylessNonlinearity(Method="Cubic polynomial", ...
%     LinearGain=gain,AMPMConversion=0,ReferenceImpedance=50);
% 
% out_memory = pa(ofdmModOut);
% 
% espectro(1024,fs,out_memory,'pwelch',ofdmModOut,'pwelch')
% return

%% PA sem memória lookup table
% paChar = pa_performance_characteristics();
% amplifier = comm.MemorylessNonlinearity('Method','Lookup table', ...
%     'Table',paChar,'ReferenceImpedance',50);
% 
% out_memory = amplifier(ofdmModOut);
% espectro(1024,fs,out_memory,'pwelch',ofdmModOut,'pwelch')
% return

%% PLOT AM/AM e AM/PM
% plot(amplifier)

% Potência em dBm
% power_x_dbm = 20*log10(abs(results.InputWaveform)) + 30 -20; % Potência do sinal de entrada
% power_y_dbm = 20*log10(abs(out_memory)) + 30 - 20; % Potência do sinal de saída
% 
% % AM/AM: Potência de saída (dBm) vs Potência de entrada (dBm)
% figure;
% scatter(power_x_dbm, power_y_dbm, 10, 'filled');
% xlabel('Potência de Entrada (dBm)');
% ylabel('Potência de Saída (dBm)');
% title('Curva AM/AM em dBm');
% % axis([10 40 20 50])
% grid on;
% 
% % Ganho
% figure;
% scatter(power_x_dbm, power_y_dbm-power_x_dbm, 10, 'filled');
% xlabel('Potência de Entrada (dBm)');
% ylabel('Ganho (dB)');
% title('Curva de Ganho');
% % axis([10 40 20 50])
% grid on;

% 
% figure
% pIndBm = 10:0.1:40;
% vin = 10.^((pIndBm-30)/20)';
% rfpa = rf.PAmemory(CoefficientMatrix=fitCoefMatMem');
% vout = rfpa(vin);
% pOutdBm = 20*log10(abs(vout))+30;
% scatter(pIndBm, pOutdBm, 10, 'filled');
% grid on;
% 
% % AM/PM: Diferença de fase vs Potência de entrada (dBm)
% phase_diff = unwrap(angle(out_memory)) - unwrap(angle(results.InputWaveform)); % Diferença de fase
% figure;
% scatter(power_x_dbm, phase_diff, 10, 'filled');
% xlabel('Potência de Entrada (dBm)');
% ylabel('Diferença de Fase (\Delta\phi = \phi_y - \phi_x)');
% title('Curva AM/PM em dBm');
% grid on;
% return

%% PA com memória
% memLen = 5;
% degLen = 5;
% 
% numDataPts = length(ofdmModOut);
% halfDataPts = round(numDataPts/2);
% 
% fitCoefMatMem = helperPACharMemPolyModel('coefficientFinder', ...
%   ofdmModOut(1:halfDataPts),paOutput(1:halfDataPts),memLen,degLen,'Memory polynomial');
% 
% paOutputFitMem = helperPACharMemPolyModel('signalGenerator', ...
%   ofdmModOut, fitCoefMatMem, 'Memory polynomial');

%% PA MANUAL
% % P = 7;
% % M = 4;
% % 
% % % Matriz de coeficientes
% % M_coeffs_HP = [ 0.9295 - 0.0001i,  0.2939 + 0.0005i, -0.1270 + 0.0034i,  0.0741 - 0.0018i;  % 1ª ordem
% %                 0.1419 - 0.0008i, -0.0735 + 0.0833i, -0.0535 + 0.0004i,  0.0908 - 0.0473i;  % 3ª ordem
% %                 0.0084 - 0.0569i, -0.4610 + 0.0274i, -0.3011 - 0.1403i, -0.0623 - 0.0269i;  % 5ª ordem
% %                 0.1774 + 0.0265i,  0.0848 + 0.0613i, -0.0362 - 0.0307i,  0.0415 + 0.0429i]; % 7ª ordem
% 
% % load('matrizCoef5x5MP.mat')
% load('matrizCoef3x3MP.mat')
% M_coeffs_MP = fitCoefMatMem;
% 
% % load('matrizCoef5x5CT.mat')
% % load('matrizCoef3x3CT.mat')
% % M_coeffs_CT = fitCoefMatMem;
% 
% % X = HP(ofdmModOut,M,P);
% % X = MP(ofdmModOut,3,3);
% % X = CT(ofdmModOut,3,3);
% % coeffs = reshape(M_coeffs_MP,[],1);
% % out_memory = X*coeffs;
% 
% out_memory = pa_manual(ofdmModOut,fitCoefMatMem,'MP',3,3);
% 
% % espectro(1024,fs,out_memory,'pwelch',out_memory1,'pwelch')
% % return

%% Sincronização
% out_memory1 = sincronize(ofdmModOut,out_memory,1,0);
out_memory = sincronize(ofdmModOut,out_memory,2,1);

% espectro(1024,fs,out_memory,'pwelch',out_memory1,'pwelch')
% return

% % Visualização dos sinais
% figure;
% subplot(2,1,1);
% plot(abs(ofdmModOut), 'b', 'DisplayName', 'Sinal original u');
% hold on;
% plot(abs(out_memory), 'r--', 'DisplayName', 'Sinal atrasado y');
% plot(t, y, 'g--', 'DisplayName', 'Sinal atrasado y');
% legend;
% title('Antes da sincronização');
% 
% subplot(2,1,2);
% plot(real(ofdmModOut), 'b', 'DisplayName', 'Sinal original u');
% hold on;
% plot(real(y_sync), 'r--', 'DisplayName', 'Sinal atrasado y (sincronizado)');
% legend;
% title('Após a sincronização');
% return

%% DPD
% estimator = comm.DPDCoefficientEstimator( ...
%     'DesiredAmplitudeGaindB',14, ...
%     'PolynomialType','Memory polynomial', ...
%     'Degree',3,'MemoryDepth',3,'Algorithm','Least squares');
% 
% coef = estimator(ofdmModOut,out_memory);
% 
% % coef = M_coeffs_MP;
% 
% dpd = comm.DPD('PolynomialType','Memory polynomial', ...
%     'Coefficients',coef);
% 
% ofdmModOut_dpd = dpd(ofdmModOut);
% 
% out_memory_dpd = rfpa(ofdmModOut_dpd);
% % X = MP(ofdmModOut_dpd, 3, 3);
% % coeffs = reshape(M_coeffs_MP.',[],1);
% % out_memory_dpd = X*coeffs;
% 
% espectro(1024,fs,out_memory,'pwelch',out_memory_dpd,'pwelch')
% return

%% Vetor de entrada
% Atrasos
 % N = length(out_memory);
 % y = out_memory;
 % yk1 = zeros(N, 1); 
 % yk2 = yk1;
 % yk3 = yk2;
 % yk4 = yk3;
 % yk5 = yk4;
 % yk1(2:N) = out_memory(1:N-1); % x(k-1)
 % yk2(3:N) = out_memory(1:N-2); % x(k-2)
 % yk3(4:N) = out_memory(1:N-3); % x(k-3)
 % yk4(5:N) = out_memory(1:N-4); % x(k-4)
 % yk5(6:N) = out_memory(1:N-5); % x(k-5)
 % 
 % in = ofdmModOut;
 % ink1 = zeros(N, 1); 
 % ink2 = ink1;
 % ink1(2:N) = ofdmModOut(1:N-1); % x(k-1)
 % ink2(3:N) = ofdmModOut(1:N-2); % x(k-2)

 %% QKRLS
 delta = 0.3;
 sigma_k = 0.05;

 loop = 2;
 u = ofdmModOut;
 out_memory_QKRLS = out_memory;

 for i = 1:loop
     [a,D, ofdmModOut_QKRLS,~,u] = QKRLS(out_memory_QKRLS.', u, ofdmModOut.', 0.9999, sigma_k, delta, 500);
    
     % u = DPD_kernel(IN, a, D, 'Gauss_complex', sigma_k);
    
     u = sincronize(ofdmModOut,u,2,1);
    
     out_memory_QKRLS = rfpa(u);
     % out_memory_QKRLS = pa_manual(u,fitCoefMatMem,'MP',3,3);
    
     out_memory_QKRLS = sincronize(u,out_memory_QKRLS,2,1);
 end

 % y = out_memory;
 % yk1 = zeros(N, 1); 
 % yk2 = yk1;
 % yk1(2:N) = out_memory_QKRLS(1:N-1); % x(k-1)
 % yk2(3:N) = out_memory_QKRLS(1:N-2); % x(k-2)
 % Y = [y yk1 yk2].';
 % 
 % [a, D, ofdmModOut_QKRLS2,~] = QKRLS(Y, u, 0.9999, sigma_k, delta, 500);
 % 
 % u2 = DPD_kernel(IN, a, D, 'Gauss_complex', sigma_k);
 % 
 % out_memory_QKRLS2 = rfpa(u2);

 %% Sincronização
 % out_memory_QKRLS2 = sincronize(u2,out_memory_QKRLS2,1,1);

 %% LMS
 % Y = [y yk1 yk2 y.*abs(y) yk1.*abs(yk1) yk2.*abs(yk2) y.*abs(y).^2 yk1.*abs(yk1).^2 yk2.*abs(yk2).^2].';%   
 Y = CT(out_memory,3,3).';
 % IN = [in ink1 ink2 in.*abs(in) ink1.*abs(ink1) ink2.*abs(ink2) in.*abs(in).^2 ink1.*abs(ink1).^2 ink2.*abs(ink2).^2].';% 
 IN = CT(ofdmModOut,3,3).';

loop = 1;
u = ofdmModOut;
out_memory_LMS = out_memory;

for i = 1:loop

    Y = CT(out_memory_LMS,3,3).';

    [~, ~, w] = LMS(Y, u, 0.1);

    % u = IN.'*w;
    % u = IN*w;
    u = (w'*IN).';

    u = sincronize(ofdmModOut, u, 2, 1);
    
    out_memory_LMS = rfpa(u);
    % out_memory_LMS = pa_manual(u,fitCoefMatMem,'MP',3,3);

    out_memory_LMS = sincronize(u, out_memory_LMS, 2, 1);
end
 
 % y = out_memory_LMS;
 % yk1 = zeros(N, 1); 
 % yk2 = yk1;
 % yk1(2:N) = out_memory_LMS(1:N-1); % x(k-1)
 % yk2(3:N) = out_memory_LMS(1:N-2); % x(k-2)
 % Y = [y yk1 yk2 y.*abs(y) yk1.*abs(yk1) yk2.*abs(yk2) y.*abs(y).^2 yk1.*abs(yk1).^2 yk2.*abs(yk2).^2].';%   
 % 
 % [~, ~, w2] = LMS(Y, u, 0.1);
 % 
 % % u2 = IN.'*w2;
 % u2 = (w2'*IN).';
 % 
 % u2 = sincronize(u, u2, 0, 1);
 % 
 % out_memory_LMS = rfpa(u2);
 % % % out_memory_LMS = pa_manual(u2,fitCoefMatMem,'MP',3,3);
 % 
 % out_memory_LMS = sincronize(u2, out_memory_LMS, 0, 1);
 % out_memory_LMS = out_memory_LMS/norm(out_memory_LMS);

 % figure;
 % plot(10*log10(mse));
 % xlabel('Iteração');
 % ylabel('Erro Médio Quadrático (dB)');
 % title('Convergência do LMS');

 % out_memory_LMS = out_memory_LMS*norm(out_memory)/norm(out_memory_LMS);
 % out_memory = out_memory*norm(out_memory_LMS)/norm(out_memory);
 % espectro(1024,fs,out_memory,'pwelch',out_memory_LMS,'pwelch')
 % return

 %% QKLMS
% 
 % Vetor de entrada
 % Y = [y yk1 yk2 ];%y.*abs(y) yk1.*abs(yk1) y.*abs(yk1) y.*abs(y).^2 yk1.*abs(yk1).^2 y.*abs(yk1).^2 yk3 yk4 yk5 y.^2 yk1.*yk2 yk1.^2 yk2.^2 yk3.^2 y.^3 yk1.^3 y.^4 y.^5].';
 % 
 % [~,~, ofdmModOut_QKLMS,~] = QKLMS(Y, ofdmModOut, 0.5, 0.8, 0.6, 500);
% 
 % Modelo PA com QKLMS
 % % Atrasos
 % N = length(ofdmModOut);
 % x = ofdmModOut;
 % xk1 = zeros(N, 1); 
 % xk2 = xk1;
 % xk3 = xk2;
 % xk4 = xk3;
 % xk5 = xk4;
 % xk1(2:N) = ofdmModOut(1:N-1); % x(k-1)
 % xk2(3:N) = ofdmModOut(1:N-2); % x(k-2)
 % xk3(4:N) = ofdmModOut(1:N-3); % x(k-3)
 % xk4(5:N) = ofdmModOut(1:N-4); % x(k-4)
 % xk5(6:N) = ofdmModOut(1:N-5); % x(k-5)
 % X = [x xk1 xk2].';% 
 % 
 % [a,~, y_PA,~] = QKLMS(X, out_memory, 0.5, 0.8, 0.6, 500);
 % 
 % espectro(1024,fs,out_memory,'pwelch',y_PA,'pwelch')
 % return
 % 
 % % [~,~, ofdmModOut_QKLMS2,~] = QKLMS(Y, ofdmModOut_QKLMS, 0.1, 0.8, 0.6, 500);
 % 
 % % out_memory_QKLMS = rfpa(ofdmModOut_QKLMS);
 % 
 % % X = MP(ofdmModOut_QKLMS, 3, 3);
 % % coeffs = reshape(M_coeffs_MP.',[],1);
 % % out_memory_QKLMS = X*coeffs;
 % 
 % % out_memory_QKLMS2 = rfpa(ofdmModOut_QKLMS2);

%% EX-KRLS
% 
 % Vetor de entrada
 % Y = [y yk1 yk2 ].';%y.*abs(y) yk1.*abs(yk1) y.*abs(yk1) y.*abs(y).^2 yk1.*abs(yk1).^2 y.*abs(yk1).^2 yk3 yk4 yk5 y.^2 yk1.*yk2 yk1.^2 yk2.^2 yk3.^2 y.^3 yk1.^3 y.^4 y.^5].';
 % 
 % [ofdmModOut_EXKRLS,~,~] = EX_KRLS1(Y,ofdmModOut,0.8,0.9999,0.01,0.995,1e-4);
 % 
 % % [ofdmModOut_EXQKRLS2,~,~] = EX_KRLS1(Y,ofdmModOut_EXKRLS,0.8,0.9999,0.01,0.995,1e-4,0.2, 500);
 % 
 % out_memory_EXKRLS = rfpa(ofdmModOut_EXKRLS);
 % 
 % % X = MP(ofdmModOut_EXKRLS, 3, 3);
 % % coeffs = reshape(M_coeffs_MP.',[],1);
 % % out_memory_EXKRLS = X*coeffs;
 % 
 % % out_memory_EXKRLS2 = rfpa(ofdmModOut_EXKRLS2);
 
 %% EX-QKRLS

 loop = 1;
 u = ofdmModOut;
 out_memory_EXQKRLS = out_memory;

 for i = 1:loop

     [~,a,D,~,u] = EX_QKRLS(out_memory_EXQKRLS.', u, ofdmModOut.',0.05,0.9999,0.01,0.995,1e-4,0.3, 500);

     % ofdmModOut_EXQKRLS = DPD_kernel(IN, a, D, "Gauss_complex2", 0.8);

     % [y_EXQKRLS,ofdmModOut_EXQKRLS,out_memory_EXQKRLS,~,~] = EX_QKRLS(X,out_memory,0.8,0.9999,0.01,0.995,1e-4,0.4, 500);

     u = sincronize(ofdmModOut,u,2,1);

     % ofdmModOut_EXQKRLS = y_EXQKRLS - ofdmModOut;

     out_memory_EXQKRLS = rfpa(u);
     % out_memory_EXQKRLS = pa_manual(u,fitCoefMatMem,'MP',3,3);

     out_memory_EXQKRLS = sincronize(u,out_memory_EXQKRLS,2,1);
 end

%% EX-QKRLS2
%  [ofdmModOut_EXQKRLS2,~,~] = EX_QKRLS(Y,ofdmModOut_EXQKRLS,0.8,0.9999,0.01,0.995,1e-4,0.2, 500);
%  out_memory_EXQKRLS2 = rfpa(ofdmModOut_EXQKRLS2);
% 
%  [c, lag] = xcorr(out_memory_EXQKRLS2, ofdmModOut);
% [~, I] = max(abs(c));
% lagDiff = lag(I);
% if lagDiff > 0
%     y_sync = [out_memory_EXQKRLS2(lagDiff+1:end)];
%     length_input = length(ofdmModOut);
%     length_output = length(y_sync);
%     y_sync = [y_sync; zeros(length_input - length_output, 1)];
% else
%     y_sync = out_memory_EXQKRLS2(1:end+lagDiff);
%     length_input = length(ofdmModOut);
%     length_output = length(y_sync);
%     y_sync = [zeros(length_input - length_output, 1); y_sync];
% end
% % out_memory = y_sync;
% 
% % Normalização
% y_sync = y_sync * norm(ofdmModOut) / norm(y_sync);
% 
% % Atraso de subsample
% D = [[y_sync(2:end); 0] [0; y_sync(1:end-1)]];
% %D = [y_sync [0; y_sync(1:end-1)]];
% coeffs = (D'*D) \ (D'*ofdmModOut);
% out_memory_EXQKRLS2 = D*coeffs;

%% Spectrum

% espectro(1024,fs,out_memory,'pwelch',out_memory_dpd,'pwelch')
% title('DPD')

espectro(1024,fs,out_memory,'pwelch',out_memory_QKRLS,'pwelch')
title('QKRLS')

% espectro(1024,fs,out_memory,'pwelch',out_memory_QKRLS2,'pwelch')
% title('QKRLS2')

% out_memory = out_memory*norm(out_memory_LMS)/norm(out_memory);
espectro(1024,fs,out_memory,'pwelch',out_memory_LMS,'pwelch')
title('LMS')

% espectro(1024,fs,out_memory,'pwelch',out_memory_QKLMS,'pwelch')
% title('QKLMS')

% espectro(1024,fs,out_memory,'pwelch',out_memory_QKLMS2,'pwelch')

% espectro(1024,fs,out_memory,'pwelch',out_memory_EXKRLS,'pwelch')
% title('EX-KRLS')

espectro(1024,fs,out_memory,'pwelch',out_memory_EXQKRLS,'pwelch')
title('EX-QKRLS')

% espectro(1024,fs,out_memory,'pwelch',out_memory_EXQKRLS2,'pwelch')
% title('EX-QKRLS2')

%% Demodulação OFDM

ofdmDemodOut = ofdmdemod(out_memory,fftLength,cycPrefLen,cycPrefLen,nullIdx,OversamplingFactor=osf);
% ofdmDemodOut_mem = ofdmdemod(out_memory_dpd,fftLength,cycPrefLen,cycPrefLen,nullIdx,OversamplingFactor=osf);
ofdmDemodOut_mem_QKRLS = ofdmdemod(out_memory_QKRLS,fftLength,cycPrefLen,cycPrefLen,nullIdx,OversamplingFactor=osf);
% ofdmDemodOut_mem_QKRLS2 = ofdmdemod(out_memory_QKRLS2,fftLength,cycPrefLen,cycPrefLen,nullIdx,OversamplingFactor=osf);
ofdmDemodOut_mem_LMS = ofdmdemod(out_memory_LMS,fftLength,cycPrefLen,cycPrefLen,nullIdx,OversamplingFactor=osf);
% ofdmDemodOut_mem_QKLMS = ofdmdemod(out_memory_QKLMS,fftLength,cycPrefLen,cycPrefLen,nullIdx,OversamplingFactor=osf);
% ofdmDemodOut_mem_QKLMS2 = ofdmdemod(out_memory_QKLMS2,fftLength,cycPrefLen,cycPrefLen,nullIdx,OversamplingFactor=osf);
% ofdmDemodOut_mem_EXKRLS = ofdmdemod(out_memory_EXKRLS,fftLength,cycPrefLen,cycPrefLen,nullIdx,OversamplingFactor=osf);
ofdmDemodOut_mem_EXQKRLS = ofdmdemod(out_memory_EXQKRLS,fftLength,cycPrefLen,cycPrefLen,nullIdx,OversamplingFactor=osf);
% ofdmDemodOut_mem_EXQKRLS2 = ofdmdemod(out_memory_EXQKRLS2,fftLength,cycPrefLen,cycPrefLen,nullIdx,OversamplingFactor=osf);

%% Scatterplot

% eqOut = ofdmDemodOut;
% eqOut = eqOut / mean(abs(eqOut(:)));
% h = scatterplot(eqOut);
% hold on
% scatterplot(qamModOut,1,0,'rx',h)
% title("Constelação sem DPD - memória")

figure
eqOut = ofdmDemodOut;
eqOut = eqOut / mean(abs(eqOut(:)));
plot(real(eqOut), imag(eqOut), 'o', 'MarkerSize', 4);
hold on
plot(real(qamModOut), imag(qamModOut), 'rx', 'MarkerSize', 4);
grid on
title("Constelação sem DPD - memória")
xlabel('Parte Real');
ylabel('Parte Imaginária');

% eqOut_mem = ofdmDemodOut_mem;
% eqOut_mem = eqOut_mem / mean(abs(eqOut_mem(:)));
% h = scatterplot(eqOut_mem);
% hold on
% scatterplot(qamModOut,1,0,'rx',h)
% title("Constelação com DPD - memória")
% figure
% eqOut_mem = ofdmDemodOut_mem;
% eqOut_mem = eqOut_mem / mean(abs(eqOut_mem(:)));
% plot(real(eqOut_mem), imag(eqOut_mem), 'o', 'MarkerSize', 4);
% hold on
% plot(real(qamModOut), imag(qamModOut), 'rx', 'MarkerSize', 4);
% grid on
% title("Constelação com DPD - memória")
% xlabel('Parte Real');
% ylabel('Parte Imaginária');

figure
eqOut_mem_QKRLS = ofdmDemodOut_mem_QKRLS;
eqOut_mem_QKRLS = eqOut_mem_QKRLS / mean(abs(eqOut_mem_QKRLS(:)));
plot(real(eqOut_mem_QKRLS), imag(eqOut_mem_QKRLS), 'o', 'MarkerSize', 4);
hold on
plot(real(qamModOut), imag(qamModOut), 'rx', 'MarkerSize', 4);
grid on
title("Constelação com QKRLS - memória")
xlabel('Parte Real');
ylabel('Parte Imaginária');

% figure
% eqOut_mem_QKRLS2 = ofdmDemodOut_mem_QKRLS2;
% eqOut_mem_QKRLS2 = eqOut_mem_QKRLS2 / mean(abs(eqOut_mem_QKRLS2(:)));
% plot(real(eqOut_mem_QKRLS2), imag(eqOut_mem_QKRLS2), 'o', 'MarkerSize', 4);
% hold on
% plot(real(qamModOut), imag(qamModOut), 'rx', 'MarkerSize', 4);
% grid on
% title("Constelação com QKRLS2 - memória")
% xlabel('Parte Real');
% ylabel('Parte Imaginária');

figure
eqOut_mem_LMS = ofdmDemodOut_mem_LMS;
eqOut_mem_LMS = eqOut_mem_LMS / mean(abs(eqOut_mem_LMS(:)));
plot(real(eqOut_mem_LMS), imag(eqOut_mem_LMS), 'o', 'MarkerSize', 4);
hold on
plot(real(qamModOut), imag(qamModOut), 'rx', 'MarkerSize', 4);
grid on
title("Constelação com LMS - memória")
xlabel('Parte Real');
ylabel('Parte Imaginária');

% eqOut_mem_QKLMS = ofdmDemodOut_mem_QKLMS;
% eqOut_mem_QKLMS = eqOut_mem_QKLMS / mean(abs(eqOut_mem_QKLMS(:)));
% h = scatterplot(eqOut_mem_QKLMS);
% hold on
% scatterplot(qamModOut,1,0,'rx',h)
% title("Constelação com QKLMS - memória")

% eqOut_mem_QKLMS2 = ofdmDemodOut_mem_QKLMS2;
% eqOut_mem_QKLMS2 = eqOut_mem_QKLMS2 / mean(abs(eqOut_mem_QKLMS2(:)));
% h = scatterplot(eqOut_mem_QKLMS2);
% hold on
% scatterplot(qamModOut,1,0,'rx',h)
% title("Constelação com QKLMS2 - memória")

% figure
% eqOut_mem = ofdmDemodOut_mem_EXKRLS;
% eqOut_mem = eqOut_mem / mean(abs(eqOut_mem(:)));
% plot(real(eqOut_mem), imag(eqOut_mem), 'o', 'MarkerSize', 4);
% hold on
% plot(real(qamModOut), imag(qamModOut), 'rx', 'MarkerSize', 4);
% grid on
% title("Constelação com EX-KRLS - memória")
% xlabel('Parte Real');
% ylabel('Parte Imaginária');

figure
eqOut_mem_EXQKRLS = ofdmDemodOut_mem_EXQKRLS;
eqOut_mem_EXQKRLS = eqOut_mem_EXQKRLS / mean(abs(eqOut_mem_EXQKRLS(:)));
plot(real(eqOut_mem_EXQKRLS), imag(eqOut_mem_EXQKRLS), 'o', 'MarkerSize', 4);
hold on
plot(real(qamModOut), imag(qamModOut), 'rx', 'MarkerSize', 4);
grid on
title("Constelação com EX-QKRLS - memória")
xlabel('Parte Real');
ylabel('Parte Imaginária');

% figure
% eqOut_mem = ofdmDemodOut_mem_EXQKRLS2;
% eqOut_mem = eqOut_mem / mean(abs(eqOut_mem(:)));
% plot(real(eqOut_mem), imag(eqOut_mem), 'o', 'MarkerSize', 4);
% hold on
% plot(real(qamModOut), imag(qamModOut), 'rx', 'MarkerSize', 4);
% grid on
% title("Constelação com EX-QKRLS - memória")
% xlabel('Parte Real');
% ylabel('Parte Imaginária');

%% EVM
% Sem DPD
error_vector = eqOut - qamModOut;
evm_percent = 100*sqrt(mean(abs(error_vector).^2)) / sqrt(mean(abs(qamModOut).^2));
fprintf('EVM: %.2f%%\n', evm_percent);

% Com DPD
error_vector = eqOut_mem_QKRLS - qamModOut;
evm_percent = 100*sqrt(mean(abs(error_vector).^2)) / sqrt(mean(abs(qamModOut).^2));
fprintf('EVM QKRLS: %.2f%%\n', evm_percent);

% error_vector = eqOut_mem_QKRLS2 - qamModOut;
% evm_percent = 100*sqrt(mean(abs(error_vector).^2)) / sqrt(mean(abs(qamModOut).^2));
% fprintf('EVM QKRLS2: %.2f%%\n', evm_percent);

error_vector = eqOut_mem_LMS - qamModOut;
evm_percent = 100*sqrt(mean(abs(error_vector).^2)) / sqrt(mean(abs(qamModOut).^2));
fprintf('EVM LMS: %.2f%%\n', evm_percent);

error_vector = eqOut_mem_EXQKRLS - qamModOut;
evm_percent = 100*sqrt(mean(abs(error_vector).^2)) / sqrt(mean(abs(qamModOut).^2));
fprintf('EVM EX-QKRLS: %.2f%%\n', evm_percent);

%% Funções
function X = HP(x, M, P)

    num_coeff = M * (P+1)/2;
    X = zeros(length(x), num_coeff);
    
    count = 1;
    for i = 1:2:P
        branch = x .* abs(x).^(i-1);
        for j = 1:M
            atraso = zeros(size(branch));
            atraso(j:end) = branch(1:end - j + 1);
            X(:, count) = atraso;
            count = count + 1;
        end
    end
end

function X = MP(x, M, P)
    num_coef = M * P;
    X = zeros(length(x), num_coef);

    count = 1;
    for i = 1:1:P
        branch = x .* abs(x).^(i-1);
        for j = 1:M
            atraso = zeros(size(branch));
            atraso(j:end) = branch(1:end - j + 1);
            X(:, count) = atraso;
            count = count + 1;
        end
    end
end

function X = CT(x, M, P)
    N = length(x);
    M_v = zeros(N, M);
    for m = 0:M-1
        M_v(:, m+1) = [zeros(m, 1); x(1:N-m)]; 
    end
      
    M_h = ones(N, 1); 
    for k = 1:P-1
        for m = 0:M-1
            X_delayed = [zeros(m, 1); x(1:N-m)];
            M_h = [M_h, abs(X_delayed).^k];
        end
    end
    
    [~, a] = size(M_h);
    X = zeros(length(x),1);
    for i = 1:a
        C = M_h(:, i) .* M_v;
        X = [X C];
    end
    
    X = X(:,2:end);

end

function paChar = pa_performance_characteristics()
    HAV08_Table =...
        [-35,60.53,0.01;
        -34,60.53,0.01;
        -33,60.53,0.08;
        -32,60.54,0.08;
        -31,60.55,0.1;
        -30,60.56,0.08;
        -29,60.57,0.14;
        -28,60.59,0.19;
        -27,60.6,0.23;
        -26,60.64,0.21;
        -25,60.69,0.28;
        -24,60.76,0.21;
        -23,60.85,0.12;
        -22,60.97,0.08;
        -21,61.12,-0.13;
        -20,61.31,-0.44;
        -19,61.52,-0.94;
        -18,61.76,-1.59;
        -17,62.01,-2.73;
        -16,62.25,-4.31;
        -15,62.47,-6.85;
        -14,62.56,-9.82;
        -13,62.47,-12.29;
        -12,62.31,-13.82;
        -11,62.2,-15.03;
        -10,62.15,-16.27;
        -9,62,-18.05;
        -8,61.53,-20.21;
        -7,60.93,-23.38;
        -6,60.2,-26.64;
        -5,59.38,-28.75];

    paChar = HAV08_Table;
    paChar(:,2) = paChar(:,1) + paChar(:,2);
end

function y = pa_manual(x,M_coeffs,tipo,M,P)

    switch tipo
        case 'MP'
            X = MP(x,M,P);
        case 'CT'
            X = CT(x,M,P);
        case 'HP'
            X = HP(x,M,P);
    end

    coeffs = reshape(M_coeffs,[],1);

    y = X*coeffs;
end