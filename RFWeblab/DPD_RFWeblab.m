clear;
%% Parâmetros
bw = 40e6;                            % Hz
symPerFrame = 1;                      % Symbols per frame
M = 16;                               % 16-QAM
osf = 3;                              % Oversampling factor
spacing = 30e3;                       % Espaçamento
numCarr = 1272;                       % Subportadoras
cycPrefLen = 144;                     % Prefixo cíclico
fftLength = 2^ceil(log2(bw/spacing)); % 2048
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

% spectrumPlot(1024,fs,ofdmModOut,'pwelch')

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

% spectrumPlot(1024,fs_up,ofdmModOut,'pwelch')
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


%% Flag do sinal que vai ser utilizado
flag = 2;

%% OFDM CTARVER

if flag == 1
    load('ofdmModOut_CTAVER.mat');
    fs_up = 200e6;
elseif flag == 2
    load('OFDM5G_5e6.mat');
end
%% Salvando a entrada 4
ofdmModOut4 = ofdmModOut;

%% Amplificador sem DPD
% Amplificador RFWebLab_PA_meas_v1_2
[y_amp, RMSout, ~, ~] = RFWebLab_PA_meas_v1_2(ofdmModOut, RMSin);

% Exibir os resultados
disp('RMSout = ')
disp(RMSout)

%% Sincronização
y_sync = syncnorm(ofdmModOut,y_amp,1,1,0);

y_sync_nodpd = y_sync;

%% Downsampling
% fs_down = fs;
% 
% [fs_num,fs_den] = rat(fs_down/fs_up);
% 
% y_sync1 = resample(y_sync,fs_num, fs_den,5,20);
% 
% % spectrumPlot(1024,fs_down,y_sync1(2:end),'pwelch')

%% OFDM Demod
if flag == 0
    % ofdmDemodOut = ofdmdemod(y_sync1(2:end),fftLength,cycPrefLen,cycPrefLen,nullIdx,OversamplingFactor=osf);
    ofdmDemodOut = ofdmdemod(y_sync_nodpd,fftLength,cycPrefLen,cycPrefLen,nullIdx,OversamplingFactor=osf);
   
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
end

%% DPD %%
%% Resetando a entrada
ofdmModOut = ofdmModOut4;
y_sync = y_sync_nodpd;

N = length(y_sync);

%% Escolhendo o algoritmo
% tipo = 1;  % comm.DPD
% tipo = 2;  % LMS MP
% tipo = 3;  % RLS Wiener-Hammerstein
% tipo = 4;  % RLS Hammerstein-Wiener
% tipo = 5;  % CTAVER
% tipo = 6;  % QKRLS
% tipo = 7;  % EX-QKRLS
% tipo = 8;  % QKLMS
% tipo = 9;  % RLS MP
% tipo = 10; % RLS CT
% tipo = 11; % EX-RLS
% tipo = 12; % RFF-EX-RLS

tipo = 10;

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

        % y_sync_dpd = syncnorm(ofdmModOut, y_amp, 1, 1, 1);

        spectrumPlot(1024,fs_up,y_amp,'pwelch',y_amp_dpd,'pwelch')
        return

    case 2
        %% LMS MP
        X = MP(ofdmModOut,5,5).';

        loop = 1;
        u = ofdmModOut;
        % u = u * norm(u) / norm(y_sync);

        y_sync_dpd = y_sync;

        for i = 1:loop

            Y = MP(y_sync_dpd,3,5).';

            [~, ~, w] = LMS(Y, u, 0.5);

            u = (w'*X).';

            u = syncnorm(ofdmModOut, u, 0, 1, 0);  
            u = u / max(abs(u));  % Normaliza amplitude
            u = u * norm(ofdmModOut) / norm(u);
            
            [y_amp,~,~,~] = RFWebLab_PA_meas_v1_2(u, RMSin);

            y_sync_dpd = syncnorm(u, y_amp, 1, 1, 0);
        end

        spectrumPlot(1024,fs_up,y_sync_dpd,'pwelch',ofdmModOut,'pwelch',y_sync_nodpd,'pwelch')
        legend('Saída c/DPD','Entrada','Saída s/DPD')
        % return

    case 3
        %% RLS Wiener-Hammerstein
        X = WH(ofdmModOut,5,3,5).';

        loop = 1;
        u = ofdmModOut;
        % u = u * norm(u) / norm(y_sync);

        y_sync_dpd = y_sync;

        for i = 1:loop

            Y = WH(y_sync_dpd,5,3,5).';

            [~, ~, w] = RLS(Y, u, 0.9999);

            u = (w'*X).';

            u = syncnorm(ofdmModOut, u, 1, 1, 0);
            u = u / max(abs(u));  % Normaliza amplitude
            u = u * norm(ofdmModOut) / norm(u);

            [y_amp,~,~,~] = RFWebLab_PA_meas_v1_2(u, RMSin);

            y_sync_dpd = syncnorm(u, y_amp, 1, 1, 0);
        end

        spectrumPlot(1024,fs_up,y_sync_dpd,'pwelch',ofdmModOut,'pwelch',y_sync_nodpd,'pwelch')
        legend('Saída c/DPD','Entrada','Saída s/DPD')
        % return

    case 4
        %% RLS Hammerstein-Wiener
        X = HW(ofdmModOut,3,5).';

        loop = 1;
        u = ofdmModOut;
        % u = u * norm(u) / norm(y_sync);

        y_sync_dpd = y_sync;

        for i = 1:loop

            Y = HW(y_sync_dpd,3,5).';

            [~, ~, w] = RLS(Y, u, 0.9999);

            u = (w'*X).';

            u = syncnorm(ofdmModOut, u, 1, 1, 0);
            u = u / max(abs(u));  % Normaliza amplitude
            u = u * norm(ofdmModOut) / norm(u);

            [y_amp,~,~,~] = RFWebLab_PA_meas_v1_2(u, RMSin);

            y_sync_dpd = syncnorm(u, y_amp, 1, 1, 0);
        end

        spectrumPlot(1024,fs_up,y_sync_dpd,'pwelch',ofdmModOut,'pwelch',y_sync_nodpd,'pwelch')
        legend('Saída c/DPD','Entrada','Saída s/DPD')
        % return

    case 5
        %% CTARVER
        M = 3; P = 5;
        X = MP(ofdmModOut,M,P).';

        loop = 3;
        u = ofdmModOut;
        y_sync_dpd = y_sync;

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

            u = u * norm(ofdmModOut) / norm(u);

            % PA
            [y_amp,~,~,~] = RFWebLab_PA_meas_v1_2(u, RMSin);
            y_sync_dpd = syncnorm(u, y_amp, 1, 1, 0);

        end

        spectrumPlot(1024,fs_up,y_sync_dpd,'pwelch',ofdmModOut,'pwelch',y_sync_nodpd,'pwelch')
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
           
            [a,D,~,~,u] = QKRLS(y_sync_dpd.', u, ofdmModOut.', 0.9999, 0.01, 1, 10000);
           
            u = syncnorm(ofdmModOut, u, 0, 1, 0);  
            u = u / max(abs(u));  % Normaliza amplitude
            u = u * norm(ofdmModOut) / norm(u);

            [y_amp,~,~,~] = RFWebLab_PA_meas_v1_2(u, RMSin);

            y_sync_dpd = syncnorm(u, y_amp, 1, 1, 0);
        end

        spectrumPlot(1024,fs_up,y_sync_dpd,'pwelch',ofdmModOut,'pwelch',y_sync_nodpd,'pwelch')
        legend('Saída c/DPD','Entrada','Saída s/DPD')

    case 7
        %% DPD EX-QKRLS

        loop = 1;
        u = ofdmModOut;
        % u = u / norm(u);
        y_sync_dpd = y_sync;

        for i = 1:loop

            [~,D,~,~,u] = EX_QKRLS(y_sync_dpd.', u, ofdmModOut.', 0.9, 1, 0.01, 0.99999, 1e-4, 0.3, 10000);

            u = syncnorm(ofdmModOut, u, 0, 1, 0);  
            u = u / max(abs(u));  % Normaliza amplitude
            u = u * norm(ofdmModOut) / norm(u);

            [y_amp,~,~,~] = RFWebLab_PA_meas_v1_2(u, RMSin);

            y_sync_dpd = syncnorm(u, y_amp, 1, 1, 0);
        end

        spectrumPlot(1024,fs_up,y_sync_dpd,'pwelch',ofdmModOut,'pwelch',y_sync_nodpd,'pwelch')
        legend('Saída c/DPD','Entrada','Saída s/DPD')


    case 8
        %% DPD QKLMS
      
        loop = 1;
        u = ofdmModOut;
        y_sync_dpd = y_sync;

        for i = 1:loop
           
            [~,~,~,~,u] = QKLMS(y_sync_dpd.', u, ofdmModOut.', 0.5, 0.5, 0.3, 1000);

            u = syncnorm(ofdmModOut, u, 0, 1, 0);  
            u = u / max(abs(u));  % Normaliza amplitude
            u = u * norm(ofdmModOut) / norm(u);

            [y_amp,~,~,~] = RFWebLab_PA_meas_v1_2(u, RMSin);

            y_sync_dpd = syncnorm(u, y_amp, 1, 1, 0);
        end

        spectrumPlot(1024,fs_up,y_sync_dpd,'pwelch',ofdmModOut,'pwelch',y_sync_nodpd,'pwelch')
        legend('Saída c/DPD','Entrada','Saída s/DPD')

    case 9
        %% RLS MP
        X = MP(ofdmModOut,3,5).';

        loop = 1;
        u = ofdmModOut;

        y_sync_dpd = y_sync;

        for i = 1:loop

            Y = MP(y_sync_dpd,3,5).';

            [~,~,w] = RLS(Y, u, 0.99999);

            u = (w'*X).';

            u = syncnorm(ofdmModOut, u, 0, 1, 0);  
            u = u / max(abs(u));  % Normaliza amplitude
            u = u * norm(ofdmModOut) / norm(u);
            
            [y_amp,~,~,~] = RFWebLab_PA_meas_v1_2(u, RMSin);

            y_sync_dpd = syncnorm(u, y_amp, 1, 1, 0);
        end

        spectrumPlot(1024,fs_up,y_sync_dpd,'pwelch',ofdmModOut,'pwelch',y_sync_nodpd,'pwelch')
        legend('Saída c/DPD','Entrada','Saída s/DPD')
        % return

        case 10
        %% RLS CT
        X = CT(ofdmModOut,3,5).';

        loop = 1;
        u = ofdmModOut;

        y_sync_dpd = y_sync;

        for i = 1:loop

            Y = CT(y_sync_dpd,3,5).';

            [~,~,w] = RLS(Y, u, 0.99999);

            u = (w'*X).';

            u = syncnorm(ofdmModOut, u, 0, 1, 0);  
            u = u / max(abs(u));  % Normaliza amplitude
            u = u * norm(ofdmModOut) / norm(u);
            
            [y_amp,~,~,~] = RFWebLab_PA_meas_v1_2(u, RMSin);

            y_sync_dpd = syncnorm(u, y_amp, 1, 1, 0);
        end

        spectrumPlot(1024,fs_up,y_sync_dpd,'pwelch',ofdmModOut,'pwelch',y_sync_nodpd,'pwelch')
        legend('Saída c/DPD','Entrada','Saída s/DPD')
        % return

    case 11
        %% EX-RLS
        X = CT(ofdmModOut,3,5).';

        loop = 1;
        u = ofdmModOut;

        y_sync_dpd = y_sync;

        for i = 1:loop

            Y = CT(y_sync_dpd,3,5).';

            [~,~,w] = EXRLS(Y, u, 0.99999, 1e-1, 1, 1, 1, 0);
            %[~,~,w] = H_EX_RLS(Y, u, 0.99999, 1e-1, 1, 1, 1, 1, 0);
            %[~,~,w] = G_EX_RLS(Y, u, 0.99999, 1e-1, 1, 1, 1, 1, 0);

            u = (w'*X).';

            u = syncnorm(ofdmModOut, u, 0, 1, 0);  
            u = u / max(abs(u));  % Normaliza amplitude
            u = u * norm(ofdmModOut) / norm(u);
            
            [y_amp,~,~,~] = RFWebLab_PA_meas_v1_2(u, RMSin);

            y_sync_dpd = syncnorm(u, y_amp, 1, 1, 0);
        end

        spectrumPlot(1024,fs_up,y_sync_dpd,'pwelch',ofdmModOut,'pwelch',y_sync_nodpd,'pwelch')
        legend('Saída c/DPD','Entrada','Saída s/DPD')
        % return

    case 12
        %% DPD RFF-EX-RLS

        loop = 1;
        u = ofdmModOut;
        % u = u / norm(u);
        y_sync_dpd = y_sync;
        X = MP(ofdmModOut,3,1).';

        for i = 1:loop
            Y = MP(y_sync_dpd,3,1).';
            
            [~,~,u,~] = EX_RFF_RLS(Y, u, X, 1, 200, 1, 0.9999, 1e-1,1e-1,1,1);

            u = syncnorm(ofdmModOut, u, 0, 1, 0);  
            u = u / max(abs(u));  % Normaliza amplitude
            u = u * norm(ofdmModOut) / norm(u);

            [y_amp,~,~,~] = RFWebLab_PA_meas_v1_2(u, RMSin);

            y_sync_dpd = syncnorm(u, y_amp, 1, 1, 0);
        end

        spectrumPlot(1024,fs_up,y_sync_dpd,'pwelch',ofdmModOut,'pwelch',y_sync_nodpd,'pwelch')
        legend('Saída c/DPD','Entrada','Saída s/DPD')

end


%% Downsample
% fs_down = fs;
% 
% [fs_num,fs_den] = rat(fs_down/fs_up);
% 
% y_sync1_dpd = resample(y_sync_dpd,fs_num, fs_den,5,20);
% 
% % spectrumPlot(1024,fs_down,y_sync1(2:end),'pwelch',y_sync1_dpd(2:end),'pwelch')

%% OFDM demod
if flag ~= 1
    % ofdmDemodOut_dpd = ofdmdemod(y_sync1_dpd(2:end),fftLength,cycPrefLen,cycPrefLen,nullIdx,OversamplingFactor=osf);
    ofdmDemodOut_dpd = ofdmdemod(y_sync_dpd,fftLength,cycPrefLen,cycPrefLen,nullIdx,OversamplingFactor=osf);
    
    %% Constelação
    figure
    eqOut_dpd = ofdmDemodOut_dpd;
    eqOut_dpd = eqOut_dpd / mean(abs(eqOut_dpd(:)));
    plot(real(eqOut), imag(eqOut), 'g+', 'MarkerSize', 4);
    hold on
    plot(real(eqOut_dpd), imag(eqOut_dpd), 'bo', 'MarkerSize', 4);
    plot(real(qamModOut), imag(qamModOut), 'rx', 'MarkerSize', 12);
    
    grid on
    title("Constelação com DPD")
    xlabel('Parte Real');
    ylabel('Parte Imaginária');
    legend('Saída s/DPD','Saída c/DPD','Entrada')
    
    error_vector = eqOut_dpd - qamModOut;
    evm_percent = 100*sqrt(mean(abs(error_vector).^2)) / sqrt(mean(abs(qamModOut).^2));
    fprintf('EVM DPD: %.2f%%\n', evm_percent);
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
    % M1 = número de atrasos do filtro de entrada
    % P  = ordem da não-linearidade
    % M2 = número de atrasos do filtro de saída
    
    N = length(x);

    % --- PRIMEIRO FILTRO LINEAR ---
    X_lin1 = zeros(N, M1);
    for m = 1:M1
        delayed = zeros(N,1);
        delayed(m:end) = x(1:end-m+1);
        X_lin1(:, m) = delayed;
    end

    % --- BLOCO NÃO LINEAR (tipo MP) ---
    X_nl = [];
    for p = 1:P
        X_nl = [X_nl, X_lin1 .* abs(X_lin1).^(p-1)];
    end

    % --- SEGUNDO FILTRO LINEAR ---
    total_branches = size(X_nl, 2);
    X = zeros(N, total_branches * M2);
    col = 1;

    for b = 1:total_branches
        branch = X_nl(:, b);
        for m = 1:M2
            delayed = zeros(N,1);
            delayed(m:end) = branch(1:end-m+1);
            X(:, col) = delayed;
            col = col + 1;
        end
    end
end


function X = HW(x, M, P)
    % --- BLOCO NÃO LINEAR DE ENTRADA (Tipo MP) ---
    X_nl = [];
    for p = 1:P
        X_nl = [X_nl, x .* abs(x).^(p-1)];
    end
    
    % --- BLOCO LINEAR (FIR) ---
    N = length(x);
    X_lin = zeros(N, M*P);
    col = 1;
    for p = 1:P
        branch = X_nl(:, p);
        for m = 1:M
            delayed = zeros(N,1);
            delayed(m:end) = branch(1:end-m+1);
            X_lin(:, col) = delayed;
            col = col + 1;
        end
    end

    % --- BLOCO NÃO LINEAR DE SAÍDA ---
    % (Aplica AM-PM ou potência não linear sobre cada coluna)
    X = [];
    for p = 1:P
        X = [X, X_lin .* abs(X_lin).^(p-1)];
    end
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

function X_volterra = volterra(x, P, M)
    x = x(:); % garante vetor coluna
    N = length(x);
    L = N - M + 1; % número de janelas válidas
    if L < 1
        error('Sinal x muito curto para a memória especificada.');
    end

    % Monta matriz de delays
    X_delay = zeros(L, M);
    for i = 1:M
        X_delay(:, i) = x(M - i + 1 : N - i + 1);
    end

    % Gera termos da série de Volterra até ordem P
    X_volterra = [];
    for p = 1:P
        idx = nchoosek(1:M, p); % combinações de posições com repetição
        for k = 1:size(idx, 1)
            term = ones(L, 1);
            for j = 1:p
                term = term .* X_delay(:, idx(k,j));
            end
            X_volterra = [X_volterra, term];
        end
    end
end


