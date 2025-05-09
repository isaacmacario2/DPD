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

%espectro(1024,fs, ofdmModOut,'pwelch')


data1 = zeros(length(ofdmModOut):2);
for i = 1:length(ofdmModOut)
    data1(i,1) = real(ofdmModOut(i));
end
for i = 1:length(ofdmModOut)
    data1(i,2) = imag(ofdmModOut(i));
end
writematrix(data1,'data1.csv')


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


%% rf.PAmemory
% load('matrizCoef3x3CT.mat')
% rfpa = rf.PAmemory(CoefficientMatrix=fitCoefMatMem);
% % rfpa = rf.PAmemory(CoefficientMatrix=M_coeffs_MP);
% % rfpa = rf.PAmemory (Model='Cross-term Memory', CoefficientMatrix=fitCoefMat);
% out_memory = rfpa(ofdmModOut);
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

%% Gerador de coeficientes
% load('PAdata.mat')
% P = 5;
% M = 5;
% 
% paInput = results.InputWaveform;
% paOutput = results.OutputWaveform;
% numDataPts = length(paInput);
% halfDataPts = round(numDataPts/2);
% fitCoefMatMem = helperPACharMemPolyModel('coefficientFinder', paInput(1:halfDataPts),paOutput(1:halfDataPts),M,P,'memPoly');

%% PA MANUAL

load('matrizCoef5x5CT.mat')
% load('matrizCoef3x3CT.mat')
M_coeffs_CT = fitCoefMatMem;

% X = HP(ofdmModOut,M,P);
% X = MP(ofdmModOut,3,3);
% X = CT(ofdmModOut,3,3);
% coeffs = reshape(M_coeffs_MP,[],1);
% out_memory = X*coeffs;

out_memory = pa_manual(ofdmModOut,fitCoefMatMem,'CT',5,5);
% out_memory = pa_manual(results.InputWaveform,fitCoefMatMem,'MP',M,P);

% espectro(1024,fs,out_memory,'pwelch',out_memory1,'pwelch')
% return

%% PLOT AM/AM e AM/PM
% % plot(amplifier)
% 
% Potência em dBm
% power_x_dbm = 20*log10(abs(results.InputWaveform)) + 30 -20; % Potência do sinal de entrada
% power_y_dbm_est = 20*log10(abs(out_memory)) + 30 - 20; % Potência do sinal de saída estimada
% power_y_dbm = 20*log10(abs(results.OutputWaveform)) + 30 - 20; % Potência do sinal de saída

% AM/AM: Potência de saída (dBm) vs Potência de entrada (dBm)
% figure;
% subplot(1,2,1)
% scatter(power_x_dbm, power_y_dbm2, 10, 'filled');
% hold on;
% scatter(power_x_dbm, power_y_dbm1, 10, 'filled');
% xlabel('Potência de Entrada (dBm)');
% ylabel('Potência de Saída (dBm)');
% title('Curva AM/AM em dBm');
% axis([0 20 20 50])
% grid on;
% 
% % Ganho
% % figure;
% subplot(1,2,2)
% scatter(power_x_dbm, power_y_dbm2-power_x_dbm, 10, 'filled');
% hold on;
% scatter(power_x_dbm, power_y_dbm1-power_x_dbm, 10, 'filled');
% xlabel('Potência de Entrada (dBm)');
% ylabel('Ganho (dB)');
% title('Curva de Ganho');
% axis([0 20 25 30])
% grid on;

% 
% 
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


%% Sincronização
% out_memory1 = sincronize(ofdmModOut,out_memory,1,0);
out_memory = sincronize(ofdmModOut,out_memory,2,1,0);

% espectro(1024,fs,out_memory,'pwelch',out_memory1,'pwelch')
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

%% DPD IDEAL
% load('PAdata.mat')
% input = ofdmModOut;%results.InputWaveform;
% output = out_memory;%results.OutputWaveform;
% N = length(input);
% 
% IN = CT(input,5,5).';
% X = CT(output,5,5).';
% d = input;
% 
% X = X - mean(X, 2); % Centralizando cada linha de X
% d = d - mean(d);    % Centralizando d
% 
% R = (1/N) * (X * X'); % Matriz de autocorrelação (N x N)
% R = R + 1e5*eye(length(R));
% 
% p = (1/N) * (X * d);  % Vetor de correlação cruzada (N x 1)
% 
% wo = R \ p;
% 
% u = (wo'*IN).';
% 
% u = sincronize(input, u, 2, 1);
% 
% out_memory_ideal = pa_manual(u,fitCoefMatMem,'CT',5,5);
% 
% out_memory_ideal = sincronize(u, out_memory_ideal, 2, 1);
% 
% espectro(1024,fs,out_memory,'pwelch',out_memory_ideal,'pwelch')
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

 loop = 1;
 u = ofdmModOut;
 out_memory_QKRLS = out_memory;

 for i = 1:loop
     [a,D, ofdmModOut_QKRLS,~,u] = QKRLS(out_memory_QKRLS.', u, ofdmModOut.', 0.9999, sigma_k, delta, 1000);
    
     % u = DPD_kernel(IN, a, D, 'Gauss_complex', sigma_k);
    
     u = sincronize(ofdmModOut,u,2,1,0);
    
     % out_memory_QKRLS = rfpa(u);
     out_memory_QKRLS = pa_manual(u,fitCoefMatMem,'CT',5,5);
    
     out_memory_QKRLS = sincronize(u,out_memory_QKRLS,2,1,0);
 end


 %% Sincronização
 % out_memory_QKRLS2 = sincronize(u2,out_memory_QKRLS2,1,1);

 %% LMS
 IN = CT(ofdmModOut,5,5).';

loop = 1;
u = ofdmModOut;
out_memory_LMS = out_memory;

for i = 1:loop

    Y = CT(out_memory_LMS,5,5).';

    [~, ~, w] = LMS(Y, u, 0.1);

    % u = IN.'*w;
    % u = IN*w;
    u = (w'*IN).';

    u = sincronize(ofdmModOut, u, 2, 1, 0);
    
    % out_memory_LMS = rfpa(u);
    out_memory_LMS = pa_manual(u,fitCoefMatMem,'CT',5,5);

    out_memory_LMS = sincronize(u, out_memory_LMS, 2, 1, 0);
end
 

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
 % % X = CT(ofdmModOut_QKLMS, 3, 3);
 % % coeffs = reshape(M_coeffs_CT.',[],1);
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

 N = length(ofdmModOut);
 in = ofdmModOut;
 ink1 = zeros(N, 1);
 ink2 = ink1;
 ink3 = ink1;
 ink4 = ink1;
 ink1(2:N) = ofdmModOut(1:N-1); % x(k-1)
 ink2(3:N) = ofdmModOut(1:N-2); % x(k-2)
 ink3(4:N) = ofdmModOut(1:N-3);
 ink3(5:N) = ofdmModOut(1:N-4);

 IN = [in ink1 ink2 ink3 ink4].';

 loop = 1;
 u = ofdmModOut;
 out_memory_EXQKRLS = out_memory;

 for i = 1:loop
    
     [~,a,D,~,u] = EX_QKRLS(out_memory_EXQKRLS.', u, ofdmModOut.',0.05,0.9999,0.01,0.995,1e-4,0.3, 10000);

     % ofdmModOut_EXQKRLS = DPD_kernel(IN, a, D, "Gauss_complex2", 0.8);

     % [y_EXQKRLS,ofdmModOut_EXQKRLS,out_memory_EXQKRLS,~,~] = EX_QKRLS(X,out_memory,0.8,0.9999,0.01,0.995,1e-4,0.4, 500);

     u = sincronize(ofdmModOut,u,2,1,0);

     % ofdmModOut_EXQKRLS = y_EXQKRLS - ofdmModOut;

     % out_memory_EXQKRLS = rfpa(u);
     out_memory_EXQKRLS = pa_manual(u,fitCoefMatMem,'CT',5,5);

     out_memory_EXQKRLS = sincronize(u,out_memory_EXQKRLS,2,1,0);
 end


%% Spectrum

% espectro(1024,fs,out_memory,'pwelch',out_memory_dpd,'pwelch')
% title('DPD')

espectro(1024,fs, out_memory_QKRLS, 'pwelch', ofdmModOut/norm(ofdmModOut),'pwelch',out_memory,'pwelch')
title('QKRLS')
legend('Saída c/DPD','Entrada','Saída s/DPD')

% espectro(1024,fs,out_memory,'pwelch',out_memory_QKRLS2,'pwelch')
% title('QKRLS2')

% out_memory = out_memory*norm(out_memory_LMS)/norm(out_memory);
espectro(1024,fs, out_memory_LMS, 'pwelch', ofdmModOut/norm(ofdmModOut),'pwelch',out_memory,'pwelch')
title('LMS')
legend('Saída c/DPD','Entrada','Saída s/DPD')

% espectro(1024,fs,out_memory,'pwelch',out_memory_QKLMS,'pwelch')
% title('QKLMS')

% espectro(1024,fs,out_memory,'pwelch',out_memory_QKLMS2,'pwelch')

% espectro(1024,fs,out_memory,'pwelch',out_memory_EXKRLS,'pwelch')
% title('EX-KRLS')

espectro(1024,fs, out_memory_EXQKRLS, 'pwelch', ofdmModOut/norm(ofdmModOut),'pwelch',out_memory,'pwelch')
title('EX-QKRLS')
legend('Saída c/DPD','Entrada','Saída s/DPD')

% espectro(1024,fs,out_memory,'pwelch',out_memory_EXQKRLS2,'pwelch')
% title('EX-QKRLS2')

%% Demodulação OFDM

ofdmDemodOut = ofdmdemod(out_memory,fftLength,cycPrefLen,cycPrefLen,nullIdx,OversamplingFactor=osf);
% ofdmDemodOut_mem = ofdmdemod(out_memory_dpd,fftLength,cycPrefLen,cycPrefLen,nullIdx,OversamplingFactor=osf);
ofdmDemodOut_mem_QKRLS = ofdmdemod(out_memory_QKRLS,fftLength,cycPrefLen,cycPrefLen,nullIdx,OversamplingFactor=osf);
ofdmDemodOut_mem_LMS = ofdmdemod(out_memory_LMS,fftLength,cycPrefLen,cycPrefLen,nullIdx,OversamplingFactor=osf);
% ofdmDemodOut_mem_QKLMS = ofdmdemod(out_memory_QKLMS,fftLength,cycPrefLen,cycPrefLen,nullIdx,OversamplingFactor=osf);
% ofdmDemodOut_mem_EXKRLS = ofdmdemod(out_memory_EXKRLS,fftLength,cycPrefLen,cycPrefLen,nullIdx,OversamplingFactor=osf);
ofdmDemodOut_mem_EXQKRLS = ofdmdemod(out_memory_EXQKRLS,fftLength,cycPrefLen,cycPrefLen,nullIdx,OversamplingFactor=osf);

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
plot(real(eqOut), imag(eqOut), 'y+', 'MarkerSize', 4);
grid on
title("Constelação com QKRLS")
xlabel('Parte Real');
ylabel('Parte Imaginária');
legend('Saída c/DPD','Entrada','Saída s/DPD')

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
plot(real(eqOut), imag(eqOut), 'y+', 'MarkerSize', 4);
grid on
title("Constelação com LMS")
xlabel('Parte Real');
ylabel('Parte Imaginária');
legend('Saída c/DPD','Entrada','Saída s/DPD')

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
plot(real(eqOut), imag(eqOut), 'y+', 'MarkerSize', 4);
grid on
title("Constelação com EX-QKRLS")
xlabel('Parte Real');
ylabel('Parte Imaginária');
legend('Saída c/DPD','Entrada','Saída s/DPD')



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
