% %% Parâmetros da modulação
% modOrder = 16;  % for 16-QAM
% bitsPerSymbol = log2(modOrder);  % modOrder = 2^bitsPerSymbol
% numSymbols =1;
% N = 1; % Loop de DPD
%
% %% Parâmetros OFDM
% numCarr = 3200;  % number of subcarriers
% cycPrefLen = 32;  % cyclic prefix length
%
% numGBCarr = numCarr/4;  % Guard band = 400
% gbLeft = 1:numGBCarr;
% gbRight = (numCarr-numGBCarr+1):numCarr;
% nullIdx = [gbLeft gbRight]';
% numDataCarr = numCarr - length(nullIdx); % Data = 1600 - 400 - 400 = 1000
% numBits = numDataCarr*bitsPerSymbol;
%
% % Amostragem do amplificador
% fs = 200e6;   % 200 MHz
%
% % Espaçamento entre subportadoras
% spacing = fs/numCarr;
%
% % Largura de banda
% bw = spacing*numDataCarr;
%
% if bw > 160e6
%     disp('Largura de banda maior que 160 MHz');
%     return;
% end
%
% %% Criando um sinal binário
% srcBits = randi([0,1],numBits,1);
%
% %% Modulação
% qamModOut = qammod(srcBits,modOrder,"InputType","bit","UnitAveragePower",true);
%
% % Diagrama de constelação
% % scatterplot(qamModOut,1,0,'yx')
%
% %% OFDM mod
% ofdmModOut = ofdmmod(qamModOut,numCarr,cycPrefLen,nullIdx);
%
% % specAn1 = spectrumAnalyzer("NumInputPorts",1,"ShowLegend",true,ChannelNames="Sinal antes do RMSin",SampleRate=fs);
% % specAn1(ofdmModOut)
%
% clear
%% Parâmetros
bw = 40e6;                            % Hz
symPerFrame = 1;                      % Symbols per frame
M = 16;                               % 16-QAM
osf = 3;                              % Oversampling factor
spacing = 30e3;                       % Espaçamento
numCarr = 1272;                       % Subportadoras
cycPrefLen = 144;                     % Prefixo cíclico
fftLength = 2^ceil(log2(bw/spacing)); % 4096
numGBCarr = fftLength - numCarr;      % Intervalo de guarda (820)
fs = spacing*fftLength*osf;           % Frequência de amostragem

numDataCarr = fftLength - numGBCarr - 1; % Data = 4096-820-1 = 3275
nullIdx = [1:numGBCarr/2+1 fftLength-numGBCarr/2+1:fftLength]';

%% Gerando sinal aleatótio
x = randi([0 M-1],numDataCarr,symPerFrame);

%% Modulação QAM
qamModOut = qammod(x,M,"UnitAveragePower",true);

%% OFDM
ofdmModOut = ofdmmod(qamModOut/osf,fftLength,cycPrefLen,nullIdx,OversamplingFactor=osf);

% espectro(1024,fs,ofdmModOut,'pwelch')

%% Salvando a entrada
ofdmModOut1 = ofdmModOut;

%% Upsampling
% fs_up = 200e6;
% 
% [fs_num,fs_den] = rat(fs_up/fs);
% 
% 
% normFc = .98 / max(fs_num,fs_den); 
% order = 256 * max(fs_num,fs_den); 
% beta = 12; 
% 
% lpFilt = firls(order, [0 normFc normFc 1],[1 1 0 0]); 
% lpFilt = lpFilt .* kaiser(order+1,beta)'; 
% lpFilt = lpFilt / sum(lpFilt); % multiplicar por p 
% lpFilt = fs_num * lpFilt; % reamostrar e plotar a resposta 
% y = resample(x,fs_num,fs_den,lpFilt); 
% 
% ofdmModOut = resample(ofdmModOut,fs_num, fs_den,lpFilt);

% fs = fs*fs_num/fs_den; % fs = fs_up

% % Filter
% upsample_rate_inverse = fs_den/fs_num;
% lpf = firls(50, [0 upsample_rate_inverse upsample_rate_inverse+0.1 1],[1 1 0 0]);
% % ofdmModOut = [ofdmModOut; zeros(100,1)];
% ofdmModOut = filter(lpf, 1, ofdmModOut);

% espectro(1024,fs_up,ofdmModOut,'pwelch')
% return

fs_up = fs;

%% Salvando a entrada 2
ofdmModOut2 = ofdmModOut;

%% MP
% X = MP(ofdmModOut,5,7).';
% [Nw, ~] = size(X);
% w = ones(Nw,1);
% w(1) = 1;
% 
% ofdmModOut = (w'*X).';

%% Salvando a entrada 3
% ofdmModOut3 = ofdmModOut;

%% Critério PA
% % Garantir que o número de amostras seja par
% totalSamples = (numCarr + cycPrefLen) * numSymbols;
% if mod(totalSamples, 2) ~= 0
%     cycPrefLen = cycPrefLen+1;  % Ajustar para número par de amostras
%     ofdmModOut = ofdmmod(qamModOut,numCarr,cycPrefLen,nullIdx);
% end
%
% % Garantir o número de amostras
% if totalSamples < 1000 || totalSamples > 1e6
%     disp('Número de amostras incompatível')
%     return;
% end

% Definir RMSin
RMSin = -22;  % dBm

% ofdmModOut = set_power(ofdmModOut,RMSin);

ofdmModOut = ofdmModOut / max(abs(ofdmModOut));

% data2 = zeros(length(ofdmModOut):2);
% for i = 1:length(ofdmModOut)
%     data2(i,1) = real(ofdmModOut(i));
% end
% for i = 1:length(ofdmModOut)
%     data2(i,2) = imag(ofdmModOut(i));
% end
% writematrix(data2,'data2.csv')

% espectro(1024,fs_up,ofdmModOut,'pwelch')

%% Flag do sinal que vai ser utilizado
flag = 0;

%% OFDM CTAVER

if flag == 1
    load('ofdmModOut_CTAVER.mat');
    fs_up = 200e6;

    data3 = zeros(length(ofdmModOut):2);
    for i = 1:length(ofdmModOut)
        data3(i,1) = real(ofdmModOut(i));
    end
    for i = 1:length(ofdmModOut)
        data3(i,2) = imag(ofdmModOut(i));
    end
    writematrix(data3,'data3.csv')
end
%% Salvando a entrada 4
ofdmModOut4 = ofdmModOut;

%% OFDM IQTools

if flag == 2
    load('ofdm_test.mat');

    ofdmModOut = Y.';

    fs = 12.5e6;
    
    % ofdmModOut = ofdmModOut / max(abs(ofdmModOut));
    % 
    % Y = ofdmModOut.'; 

    % espectro(1024,fs,ofdmModOut,'pwelch')

    % data5 = zeros(length(ofdmModOut):2);
    % for i = 1:length(ofdmModOut)
    %     data5(i,1) = real(ofdmModOut(i));
    % end
    % for i = 1:length(ofdmModOut)
    %     data5(i,2) = imag(ofdmModOut(i));
    % end
    % writematrix(data5,'data5.csv')
end

if flag == 3
    load('ofdm_std.mat');

    ofdmModOut = Y.';

    fs = 12.5e6;
end

if flag == 4
    load('ofdm_1024.mat');

    ofdmModOut = Y.';

    fs = 12.5e6;
end

%% Salvando a entrada 5
ofdmModOut5 = ofdmModOut;

%% Amplificador sem DPD
% Amplificador RFWebLab_PA_meas_v1_2
[y_amp, RMSout, ~, ~] = RFWebLab_PA_meas_v1_2(ofdmModOut, RMSin);

% Exibir os resultados
disp('RMSout = ')
disp(RMSout)

%% Sincronização
y_sync = sincronize(ofdmModOut,y_amp,1,1,0);

%  figure
%  plot(real(ofdmModOut), 'b', 'DisplayName', 'Sinal original u');
% hold on;
% plot(real(y_sync), 'r--', 'DisplayName', 'Sinal atrasado y (sincronizado)');
% legend;
% title('Após a sincronização');

% signal_energy = norm(y_sync)^2; % Energia total do sinal
% avg_energy = signal_energy / length(y_sync); % Energia média dividindo pela quantidade de amostras
% power_watts = avg_energy / 50; % Ajusta para impedância de 50 ohms
% rms_power = 10 * log10(power_watts) + 30; % Converte a potência de watts para dbm
%
% if abs(RMSout - rms_power) > 0.1
%    disp('RMS errado');
%    return;
% end

% espectro(1024,fs_up,y_sync,'pwelch',ofdmModOut,'pwelch')
% return

y_sync_nodpd = y_sync;

%% Downsampling
% fs_down = fs;
% 
% [fs_num,fs_den] = rat(fs_down/fs_up);
% 
% y_sync1 = resample(y_sync,fs_num, fs_den,5,20);
% 
% % espectro(1024,fs_down,y_sync1(2:end),'pwelch')

%% OFDM Demod
if flag == 0
    % ofdmDemodOut = ofdmdemod(y_sync1(2:end),fftLength,cycPrefLen,cycPrefLen,nullIdx,OversamplingFactor=osf);
    ofdmDemodOut = ofdmdemod(y_sync,fftLength,cycPrefLen,cycPrefLen,nullIdx,OversamplingFactor=osf);
    
    %% Constelação
    % figure
    eqOut = ofdmDemodOut;
    eqOut = eqOut / mean(abs(eqOut(:)));
    % plot(real(eqOut), imag(eqOut), 'o', 'MarkerSize', 4);
    % hold on
    % plot(real(qamModOut), imag(qamModOut), 'rx', 'MarkerSize', 4);
    % grid on
    % title("Constelação sem DPD")
    % xlabel('Parte Real');
    % ylabel('Parte Imaginária');
    error_vector = eqOut - qamModOut;
    evm_percent = 100*sqrt(mean(abs(error_vector).^2)) / sqrt(mean(abs(qamModOut).^2));
    fprintf('EVM: %.2f%%\n', evm_percent);

% eqOut = ofdmDemodOut;
% eqOut = eqOut / mean(abs(eqOut(:)));
% h = scatterplot(eqOut);
% hold on
% scatterplot(qamModOut,1,0,'rx',h)
% title("Constelação sem DPD")

% ref = qamModOut;
% [scaledSig, gain] = scaleRotate(ofdmDemodOut,ref);
%
% eqOut = scaledSig;
% eqOut = eqOut / mean(abs(eqOut(:)));
% h = scatterplot(eqOut);
% hold on
% scatterplot(qamModOut,1,0,'rx',h)
% title("Constelação sem DPD")
end

%% DPD %%
%% Resetando a entrada
ofdmModOut = ofdmModOut4;
y_sync = y_sync_nodpd;

N = length(y_sync);

% in = ofdmModOut;
% ink1 = zeros(N, 1); 
% ink2 = ink1;
% ink3 = ink1;
% ink4 = ink1;
% ink5 = ink1;
% ink1(2:N) = ofdmModOut(1:N-1); % x(k-1)
% ink2(3:N) = ofdmModOut(1:N-2); % x(k-2)
% ink3(4:N) = ofdmModOut(1:N-3);
% ink4(5:N) = ofdmModOut(1:N-4);
% ink5(6:N) = ofdmModOut(1:N-5);

%% Escolhendo o algoritmo
% tipo = 1;  % comm.DPD
% tipo = 2;  % LMS MP
% tipo = 3;  % LMS Wiener
% tipo = 4;  % LMS Bessel
% tipo = 5;  % CTAVER
% tipo = 6;  % QKRLS
% tipo = 7;  % EX-QKRLS
% tipo = 8;  % QKLMS

tipo = 8;

switch tipo
    case 1
        %% DPD Coeficientes
        estimator = comm.DPDCoefficientEstimator( ...
            'DesiredAmplitudeGaindB',7, ...
            'PolynomialType','Cross-term memory polynomial', ...
            'Degree',3,'MemoryDepth',5,'Algorithm','Least squares');

        coef = estimator(ofdmModOut,y_amp);

        dpd = comm.DPD('PolynomialType','Memory polynomial','Coefficients',coef);
        
        ofdmModOut_dpd = dpd(ofdmModOut);
   
        [y_amp_dpd,~,~,~] = RFWebLab_PA_meas_v1_2(ofdmModOut_dpd, RMSin);

        % y_sync_dpd = sincronize(ofdmModOut, y_amp, 1, 1, 1);

        espectro(1024,fs_up,y_amp,'pwelch',y_amp_dpd,'pwelch')
        return

    case 2
        %% LMS MP
        % Y = [y yk1 yk2 y.*abs(y) yk1.*abs(yk1) yk2.*abs(yk2) y.*abs(y).^2 yk1.*abs(yk1).^2 yk2.*abs(yk2).^2].';%
        % X = [in ink1 ink2 in.*abs(in) ink1.*abs(ink1) ink2.*abs(ink2) in.*abs(in).^2 ink1.*abs(ink1).^2 ink2.*abs(ink2).^2].';%
        X = MP(ofdmModOut,5,5).';

        loop = 1;
        u = ofdmModOut;
        % u = u * norm(u) / norm(y_sync);

        y_sync_dpd = y_sync;

        for i = 1:loop

            Y = MP(y_sync_dpd,5,5).';

            [~, ~, w] = LMS(Y, u, 0.5);

            % u = X.'*w;
            % u = X*w;
            u = (w'*X).';

            u = sincronize(ofdmModOut, u, 0, 1, 0);  
            u = u / max(abs(u));  % Normaliza amplitude
            u = u * norm(ofdmModOut) / norm(u);
            
            [y_amp,~,~,~] = RFWebLab_PA_meas_v1_2(u, RMSin);

            y_sync_dpd = sincronize(u, y_amp, 1, 1, 0);
        end

        espectro(1024,fs_up,y_sync_dpd,'pwelch',ofdmModOut,'pwelch',y_sync_nodpd,'pwelch')
        legend('Saída c/DPD','Entrada','Saída s/DPD')
        % return

    case 3
        %% LMS Wiener
        X = WH(ofdmModOut,5,3,5).';

        loop = 1;
        u = ofdmModOut;
        % u = u * norm(u) / norm(y_sync);

        y_sync_dpd = y_sync;

        for i = 1:loop

            Y = WH(y_sync_dpd,5,3,5).';

            [~, ~, w] = LMS(Y, u, 0.05);

            % u = X.'*w;
            % u = X*w;
            u = (w'*X).';

            u = sincronize(ofdmModOut, u, 1, 1, 0);

            % u = set_power(u,RMSin);

            % u_pow = sincronize(ofdmModOut, u_pow, 2, 1 ,0);

            [y_amp,~,~,~] = RFWebLab_PA_meas_v1_2(u, RMSin);

            y_sync_dpd = sincronize(u, y_amp, 1, 1, 0);
        end

        espectro(1024,fs_up,y_sync_dpd,'pwelch',ofdmModOut,'pwelch',y_sync_nodpd,'pwelch')
        legend('Saída c/DPD','Entrada','Saída s/DPD')
        % return

    case 4
        %% LMS Bessel
        X = bessel(ofdmModOut,5,4).';

        loop = 1;
        u = ofdmModOut;
        % u = u * norm(u) / norm(y_sync);

        y_sync_dpd = y_sync;

        for i = 1:loop

            Y = bessel(y_sync_dpd,5,4).';

            [~, ~, w] = LMS(Y, u, 0.05);

            % u = X.'*w;
            % u = X*w;
            u = (w'*X).';

            u = sincronize(ofdmModOut, u, 1, 1, 0);

            % u = set_power(u,RMSin);

            % u_pow = sincronize(ofdmModOut, u_pow, 2, 1 ,0);

            [y_amp,~,~,~] = RFWebLab_PA_meas_v1_2(u, RMSin);

            y_sync_dpd = sincronize(u, y_amp, 1, 1, 0);
        end

        espectro(1024,fs_up,y_sync_nodpd,'pwelch',y_sync_dpd,'pwelch')
        % return

    case 5
        %% CTAVER
        M = 3; P = 5;
        X = MP(ofdmModOut,M,P).';

        loop = 3;
        u = ofdmModOut;
        y_sync_dpd = y_sync;

        % % u = u * norm(u) / norm(y_sync);
        % 
        % y_sync_dpd = y_sync;

        % Iniciando o vetor de coeficientes
        [Nw, N] = size(X);
        w = zeros(Nw,1);
        w(1) = 1;
        
        for i = 1:loop

            Y = MP(y_sync_dpd,M,P).';
          
            % Newton
            % z = (w'*Y).';
            % error = u - z;                         % e = u - z
            % ls_result = ls_estimation(Y.', error); % beta = (Y'Y + lambda*I)^-1 * (Y'*e)
            % w = w + 0.9*ls_result;                 % w = w + mu*beta

            % LMS
            % [~, ~, w] = LMS(Y, u, 0.1);
            z = (w'*Y).';
            error = u - z;
            w = w + 0.1*(Y*error);

            % Predistort
            X = MP(ofdmModOut,M,P).';
            u = (w'*X).';

            % u = sincronize(ofdmModOut, u, 0, 1, 0);  
            % u = set_power(u,RMSin);
            % u = u / max(abs(u));  % Normaliza amplitude
            u = u * norm(ofdmModOut) / norm(u);


            % PA
            [y_amp,~,~,~] = RFWebLab_PA_meas_v1_2(u, RMSin);
            y_sync_dpd = sincronize(u, y_amp, 1, 1, 0);

        end

       
        espectro(1024,fs_up,y_sync_dpd,'pwelch',ofdmModOut,'pwelch',y_sync_nodpd,'pwelch')
        legend('Saída c/DPD','Entrada','Saída s/DPD')
        return

    case 6
        %% DPD QKRLS
        % Vetor de entrada

        % X = [in ink1 ink2 ink3 ink4 ink5].';%

        loop = 1;
        u = ofdmModOut;
        y_sync_dpd = y_sync;

        for i = 1:loop
            % y = y_sync_dpd;
            % yk1 = zeros(N, 1);
            % yk2 = yk1;
            % yk3 = yk1;
            % yk4 = yk1;
            % yk5 = yk1;
            % yk1(2:N) = y_sync_dpd(1:N-1); % x(k-1)
            % yk2(3:N) = y_sync_dpd(1:N-2); % x(k-2)
            % yk3(4:N) = y_sync_dpd(1:N-3); % x(k-3)
            % yk4(5:N) = y_sync_dpd(1:N-4); % x(k-3)
            % yk5(6:N) = y_sync_dpd(1:N-5); % x(k-3)
            % 
            % Y = [y yk1 yk2 yk3 yk4 yk5].';

            [a,D,~,~,u] = QKRLS(y_sync_dpd.', u, ofdmModOut.', 0.9999, 0.05, 0.3, 10000);
            % [a,D,~,~,u] = QKRLS(Y, u, X, 0.9999, 0.05, 0.3, 10000);

            % u = DPD_kernel(X, a, D, 'Gauss_complex', 0.05);

            u = sincronize(ofdmModOut, u, 0, 1, 0);  
            % u = set_power(u,RMSin);
            u = u / max(abs(u));  % Normaliza amplitude
            u = u * norm(ofdmModOut) / norm(u);

            [y_amp,~,~,~] = RFWebLab_PA_meas_v1_2(u, RMSin);

            y_sync_dpd = sincronize(u, y_amp, 1, 1, 0);
        end

        espectro(1024,fs_up,y_sync_dpd,'pwelch',ofdmModOut,'pwelch',y_sync_nodpd,'pwelch')
        legend('Saída c/DPD','Entrada','Saída s/DPD')

    case 7
        %% DPD EX-QKRLS

        % X = [in ink1 ink2 ink3].';% ink4 ink5
        % X = MP(ofdmModOut,5,7).';

        % [ofdmModOut_dpd,~,~] = EX_QKRLS(Y,ofdmModOut,0.8,0.9999,0.01,0.995,1e-4,0.6, 500);
        % N = length(ofdmModOut_dpd);
        % ofdmModOut_dpd(1:N-2) = ofdmModOut_dpd(3:N);

        loop = 1;
        u = ofdmModOut;
        % u = u / norm(u);
        y_sync_dpd = y_sync;

        for i = 1:loop

            % y = y_sync_dpd;
            % yk1 = zeros(N, 1);
            % yk2 = yk1;
            % yk3 = yk1;
            % yk4 = yk1;
            % yk5 = yk1;
            % yk1(2:N) = y_sync_dpd(1:N-1); % x(k-1)
            % yk2(3:N) = y_sync_dpd(1:N-2); % x(k-2)
            % yk3(4:N) = y_sync_dpd(1:N-3); % x(k-3)
            % yk4(5:N) = y_sync_dpd(1:N-4); % x(k-3)
            % yk5(6:N) = y_sync_dpd(1:N-5); % x(k-3)

            % Y = [y yk1 yk2 yk3].';%; yk4 yk5
            % Y = MP(y_sync_dpd,5,7).';


            [~,D,~,~,u] = EX_QKRLS(y_sync_dpd.', u, ofdmModOut.', 0.9, 1, 0.01, 0.995, 1e-4, 0.3, 10000);
            % [~,D,~,~,u] = EX_QKRLS(Y, u, X, 0.9, 1, 0.01, 0.995, 1e-4, 0.8, 10000);

            u = sincronize(ofdmModOut, u, 0, 1, 0);  
            u = u / max(abs(u));  % Normaliza amplitude
            u = u * norm(ofdmModOut) / norm(u);

            [y_amp,~,~,~] = RFWebLab_PA_meas_v1_2(u, RMSin);

            y_sync_dpd = sincronize(u, y_amp, 1, 1, 0);
        end

        espectro(1024,fs_up,y_sync_dpd,'pwelch',ofdmModOut,'pwelch',y_sync_nodpd,'pwelch')
        legend('Saída c/DPD','Entrada','Saída s/DPD')

        % MSEdB = 10*log10(mse);
        % 
        % figure
        % plot(MSEdB, 'g', 'linewidth', 1)
        % grid on;
        % xlabel('k')
        % ylabel('\xi(k) [dB]')
        % title('EX-QKRLS')

    case 8
        %% DPD QKLMS
        %Vetor de entrada
        % Y = [y yk1 yk2];
        %
        % 
        loop = 1;
        u = ofdmModOut;
        y_sync_dpd = y_sync;

        for i = 1:loop
           
            [~,~,~,~,u] = QKLMS(y_sync_dpd.', u, ofdmModOut.', 0.5, 0.5, 0.3, 1000);

            % u = DPD_kernel(X, a, D, 'Gauss_complex', 0.05);

            u = sincronize(ofdmModOut, u, 0, 1, 0);  
            % u = set_power(u,RMSin);
            u = u / max(abs(u));  % Normaliza amplitude
            u = u * norm(ofdmModOut) / norm(u);

            [y_amp,~,~,~] = RFWebLab_PA_meas_v1_2(u, RMSin);

            y_sync_dpd = sincronize(u, y_amp, 1, 1, 0);
        end

        espectro(1024,fs_up,y_sync_dpd,'pwelch',ofdmModOut,'pwelch',y_sync_nodpd,'pwelch')
        legend('Saída c/DPD','Entrada','Saída s/DPD')

end

%% DPD EX-KRLS
% Y = [y yk1 yk2].';
% 
% [ofdmModOut_dpd,~,~] = EX_KRLS1(Y,ofdmModOut,0.8,0.9999,0.01,0.995,1e-4);

%% Downsample
% fs_down = fs;
% 
% [fs_num,fs_den] = rat(fs_down/fs_up);
% 
% y_sync1_dpd = resample(y_sync_dpd,fs_num, fs_den,5,20);
% 
% % espectro(1024,fs_down,y_sync1(2:end),'pwelch',y_sync1_dpd(2:end),'pwelch')

%% OFDM demod
if flag ~= 1
    % ofdmDemodOut_dpd = ofdmdemod(y_sync1_dpd(2:end),fftLength,cycPrefLen,cycPrefLen,nullIdx,OversamplingFactor=osf);
    ofdmDemodOut_dpd = ofdmdemod(y_sync_dpd,fftLength,cycPrefLen,cycPrefLen,nullIdx,OversamplingFactor=osf);
    
    %% Constelação
    figure
    eqOut_dpd = ofdmDemodOut_dpd;
    eqOut_dpd = eqOut_dpd / mean(abs(eqOut_dpd(:)));
    plot(real(eqOut_dpd), imag(eqOut_dpd), 'o', 'MarkerSize', 4);
    hold on
    plot(real(qamModOut), imag(qamModOut), 'rx', 'MarkerSize', 4);
    plot(real(eqOut), imag(eqOut), 'y+', 'MarkerSize', 4);
    grid on
    title("Constelação com DPD")
    xlabel('Parte Real');
    ylabel('Parte Imaginária');
    legend('Saída c/DPD','Entrada','Saída s/DPD')
    
    error_vector = eqOut_dpd - qamModOut;
    evm_percent = 100*sqrt(mean(abs(error_vector).^2)) / sqrt(mean(abs(qamModOut).^2));
    fprintf('EVM DPD: %.2f%%\n', evm_percent);
end

% eqOut = ofdmDemodOut_dpd;
% eqOut = eqOut / mean(abs(eqOut(:)));
% h = scatterplot(eqOut);
% hold on
% scatterplot(qamModOut,1,0,'rx',h)
% title("Constelação com DPD")

% ref = qamModOut;
% [scaledSig, gain] = scaleRotate(ofdmDemodOut_dpd,ref);
% 
% eqOut = scaledSig;
% eqOut = eqOut / mean(abs(eqOut(:)));
% h = scatterplot(eqOut);
% hold on
% scatterplot(qamModOut,1,0,'rx',h)
% title("Constelação com DPD")

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

function X = wiener(x, M, P)
% WienerDPD: Gera a matriz de regressão para o modelo de Wiener.
% x - Sinal de entrada
% M - Ordem do filtro FIR (memória)
% P - Ordem do polinômio (não linearidade)

    % Filtro FIR linear
    h = fir1(M-1, 0.5);  % Filtro FIR simples (ajustável)
    x_filt = filter(h, 1, x);  % Aplica o filtro FIR
    
    % Inicializa matriz de regressão
    num_coeff = P;
    X = zeros(length(x_filt), num_coeff);

    % Aplica a não linearidade (polinômio)
    for p = 1:P
        X(:, p) = x_filt .* abs(x_filt).^(p-1);
    end
end

function X = bessel(x, M, P)

    X = zeros(length(x), P*M);
    count = 1;
    for delay = 0:M-1
        x_shifted = [zeros(delay,1); x(1:end-delay)];
        for n = 0:P-1
            X(:, count) = besselj(n, abs(x_shifted));
            count = count + 1;
        end
    end

end

function X = MODF(x, M, P)
% MODF - Gera a matriz de regressão para DPD usando o Filtro de Diferença de Múltiplas Ordens
% x: sinal de entrada
% M: número de atrasos de memória
% P: ordem máxima de diferença

    N = length(x);
    X = zeros(N, M*P);
    count = 1;

    for p = 1:P
        x_diff = diff([zeros(p,1); x], p);  % Diferença de ordem p
        x_diff = x_diff / norm(x_diff);      % Normaliza cada diferença

        for m = 0:M-1
            atraso = [zeros(m,1); x_diff(1:N-m)];
            X(:, count) = atraso;
            count = count + 1;
        end
    end

    % Normaliza a matriz completa
    X = X / norm(X,'fro');
end

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

function X = WH(x, M1, P, M2)
% x: Sinal de entrada
% M1: Ordem do primeiro filtro FIR
% P: Ordem do polinômio não linear
% M2: Ordem do segundo filtro FIR

    % Primeiro filtro FIR
    h1 = fir1(M1-1, 0.4); 
    x_filt1 = filter(h1, 1, x);

    % Bloco não linear
    X_nl = zeros(length(x_filt1), P);
    for p = 1:P
        X_nl(:, p) = x_filt1 .* abs(x_filt1).^(p-1);
    end

    % Segundo filtro FIR
    h2 = fir1(M2-1, 0.4);
    X = filter(h2, 1, X_nl);

end

function beta = ls_estimation(X, y)
    %ls_estimation
    % Solves problems where we want to minimize the error between a
    % lienar model and some input/output data.
    %
    %     min || y - X*beta ||^2
    %
    % A small regularlizer, lambda, is included to improve the
    % conditioning of the matrix.
    %
    
    % Trim X and y to get rid of 0s in X.
    % X = X(5:end, :);
    % y = y(5:end);
    
    lambda = 0.001;
    beta = (X'*X + lambda*eye(size((X'*X)))) \ (X'*y);
end

 