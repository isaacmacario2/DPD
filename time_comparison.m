%% Dados de exemplo
num_samples = 1000;          % Número de amostras
sigma2_x = 0.1;               % Variância do sinal de entrada
sigma2_n = 0.001;             % Variância do ruído de observação
ENS = 100;                    % Ensemble  

MSEexqkrls1 = zeros(num_samples,1);  % MSE ex-QKRLS
MSEexqkrls2 = zeros(num_samples,1);  % MSE ex-QKRLS
MSEexqkrls3 = zeros(num_samples,1);  % MSE ex-QKRLS
MSEexkrls = zeros(num_samples,1);  % MSE ex-KRLS1

lambda = 0.9999;                % Fator de esquecimento
sigma_k = 0.5;                % Largura do kernel
delta_q1 = 0.4;                % Tamanho da quantização
delta_q2 = 0.7;                % Tamanho da quantização
delta_q3 = 1;                % Tamanho da quantização

Lmax = 1000;                    % Tamanho máximo do dicionário
                  
t_stop = zeros(ENS,1);
t_stop1 = zeros(ENS,1);
t_stop2 = zeros(ENS,1);
t_stop3 = zeros(ENS,1);


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
    X = [x xk1 xk2].';
    
    % Vetor desejado
    d = -.76*x - xk1 + xk2 + .5*x.^2 + 2*x.*xk2 - 1.6*xk1.^2 + 1.2*xk2.^2 + .8*xk1.*xk2 + x.^3 + n;
   
   
    % EX_KRLS
    t_start = tic;
    [~,mse_ex] = EX_KRLS(X,d,"Gauss_complex2",sigma_k,1,0.01,0.9999,1e-4);
    t_stop(i,:) = toc(t_start);

    % EX_QKRLS1
    t_start1 = tic;
    [~,~,~,mse_ex1,~] = EX_QKRLS(X,d,0,sigma_k,1,0.01,0.9999,1e-4,delta_q1, Lmax);
    t_stop1(i,:) = toc(t_start1);

    % EX_QKRLS2
    t_start2 = tic;
    [~,~,~,mse_ex2,~] = EX_QKRLS(X,d,0,sigma_k,1,0.01,0.9999,1e-4,delta_q2, Lmax);
    t_stop2(i,:) = toc(t_start2);

    % EX_QKRLS3
    t_start3 = tic;
    [~,~,~,mse_ex3,~] = EX_QKRLS(X,d,0,sigma_k,1,0.01,0.9999,1e-4,delta_q3, Lmax);
    t_stop3(i,:) = toc(t_start3);

    MSEexkrls = MSEexkrls + mse_ex;
    MSEexqkrls1 = MSEexqkrls1 + mse_ex1;
    MSEexqkrls2 = MSEexqkrls2 + mse_ex2;
    MSEexqkrls3 = MSEexqkrls3 + mse_ex3;

end

MSEexkrls = MSEexkrls/ENS;
MSEexkrlsdB = 10*log10(MSEexkrls);

MSEexqkrls1 = MSEexqkrls1/ENS;
MSEexqkrlsdB1 = 10*log10(MSEexqkrls1);
MSEexqkrls2 = MSEexqkrls2/ENS;
MSEexqkrlsdB2 = 10*log10(MSEexqkrls2);
MSEexqkrls3 = MSEexqkrls3/ENS;
MSEexqkrlsdB3 = 10*log10(MSEexqkrls3);



time_EXKRLS = sum(t_stop)/ENS;
time_EXQKRLS1 = sum(t_stop1)/ENS;
time_EXQKRLS2 = sum(t_stop2)/ENS;
time_EXQKRLS3 = sum(t_stop3)/ENS;

fprintf('Tempo médio de execução do EX-KRLS: %f \n', time_EXKRLS )
fprintf('Tempo médio de execução do EX-QKRLS com delta = %.1f: %.3f \n', delta_q1, time_EXQKRLS1 )
fprintf('Tempo médio de execução do EX-QKRLS com delta = %.1f: %.3f \n', delta_q2, time_EXQKRLS2 )
fprintf('Tempo médio de execução do EX-QKRLS com delta = %.1f: %.3f \n', delta_q3, time_EXQKRLS3 )

MAE_diff_dB1 = mean(abs(MSEexqkrlsdB1 - MSEexkrlsdB));
MAE_diff_dB2 = mean(abs(MSEexqkrlsdB2 - MSEexkrlsdB));
MAE_diff_dB3 = mean(abs(MSEexqkrlsdB3 - MSEexkrlsdB));

fprintf('Diferença absoluta média (dB) com delta = %.1f: %.3f dB\n',delta_q1, MAE_diff_dB1);
fprintf('Diferença absoluta média (dB) com delta = %.1f: %.3f dB\n',delta_q2, MAE_diff_dB2);
fprintf('Diferença absoluta média (dB) com delta = %.1f: %.3f dB\n',delta_q3, MAE_diff_dB3);


%% Plots

% Curva de aprendizado
figure
plot(10 * log10(sigma2_n) * ones(num_samples, 1), '--b', 'linewidth', 2)
hold on
plot(MSEexkrlsdB, 'b-', 'linewidth', 1)
plot(MSEexqkrlsdB1, 'g-.', 'linewidth', 1)
plot(MSEexqkrlsdB2, 'r-.', 'linewidth', 1)
plot(MSEexqkrlsdB3, 'm-.', 'linewidth', 1)
legend('Ruído','EX-KRLS','EX-QKRLS \delta = 0.4','EX-QKRLS \delta = 0.7','EX-QKRLS \delta = 1')
grid on;
xlabel('k')
ylabel('\xi(k) [dB]')


