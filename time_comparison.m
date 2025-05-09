%% Dados de exemplo
num_samples = 1000;          % Número de amostras
sigma2_x = 0.1;               % Variância do sinal de entrada
sigma2_n = 0.001;             % Variância do ruído de observação
ENS = 100;                    % Ensemble  

MSEexqkrls = zeros(num_samples,1);  % MSE ex-QKRLS
MSEexkrls1 = zeros(num_samples,1);  % MSE ex-KRLS1
lambda = 0.9999;                % Fator de esquecimento
sigma_k = 0.5;                % Largura do kernel
delta_q = 1;                % Tamanho da quantização
Lmax = 1000;                    % Tamanho máximo do dicionário
                  
t_stop = zeros(ENS,1);
t_stop1 = zeros(ENS,1);

%% Loop principal
for i = 1:ENS
    if mod(i,1) == 0
        disp(i)
    end

    % x = randn(num_samples, 1); x = x - mean(x); x = sqrt(sigma2_x)*x/std(x); % Sinal de entrada
    % n = randn(num_samples,1); n = n - mean(n); n = sqrt(sigma2_n)*n/std(n);  % Ruído

    x = (randn(num_samples, 1) + 1j * randn(num_samples, 1)); % Sinal de entrada complexo
    x = x - mean(x); x = sqrt(sigma2_x) * x / std(x);

    n = (randn(num_samples, 1) + 1j * randn(num_samples, 1)); % Ruído complexo
    n = n - mean(n); n = sqrt(sigma2_n) * n / std(n);

    % Atrasos
    xk1 = zeros(num_samples, 1); 
    xk2 = xk1;       
    xk1(2:num_samples) = x(1:num_samples-1); % x(k-1)
    xk2(3:num_samples) = x(1:num_samples-2); % x(k-2)

    % Vetor de entrada
    X = [x xk1 xk2];
    
    % Vetor desejado
    d = -.76*x - xk1 + xk2 + .5*x.^2 + 2*x.*xk2 - 1.6*xk1.^2 + 1.2*xk2.^2 + .8*xk1.*xk2 + x.^3 + n;
   
    % EX_QKRLS
    X = X.';
    t_start = tic;
    [~,~,~,mse_ex,~] = EX_QKRLS(X,d,0,sigma_k,1,0.01,0.9999,1e-4,delta_q, Lmax);
    t_stop(i,:) = toc(t_start);

    % EX_KRLS
    % X = X.';
    t_start1 = tic;
    [~,mse_ex1] = EX_KRLS(X,d,"Gauss_complex2",sigma_k,1,0.01,0.9999,1e-4);
    t_stop1(i,:) = toc(t_start1);

  
    MSEexqkrls = MSEexqkrls + mse_ex;
    MSEexkrls1 = MSEexkrls1 + mse_ex1;

end


MSEexqkrls = MSEexqkrls/ENS;
MSEexqkrlsdB = 10*log10(MSEexqkrls);

MSEexkrls1 = MSEexkrls1/ENS;
MSEexkrls1dB = 10*log10(MSEexkrls1);

time_EXQKRLS = sum(t_stop)/ENS;
time_EXKRLS = sum(t_stop1)/ENS;

fprintf('Tempo médio de execução do EX-QKRLS: %f \n', time_EXQKRLS )
fprintf('Tempo médio de execução do EX-KRLS: %f \n', time_EXKRLS )

MAE_diff_dB = mean(abs(MSEexqkrlsdB - MSEexkrls1dB));
fprintf('Diferença absoluta média (dB): %.4f dB\n', MAE_diff_dB);

%% Plots

% Curva de aprendizado
figure
plot(10 * log10(sigma2_n) * ones(num_samples, 1), '--b', 'linewidth', 2)
hold on
plot(MSEexqkrlsdB, 'r-.', 'linewidth', 1)
plot(MSEexkrls1dB, 'g-', 'linewidth', 1)
grid on;
xlabel('k')
ylabel('\xi(k) [dB]')


