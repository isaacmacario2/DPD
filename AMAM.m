%% Sinais
load('espectros.mat')

x = ofdmModOut_geral;
y = y_sync_nodpd_geral;
y_kernel = y_sync_dpd_kernel;

power_x_dbm = 20*log10(abs(x)) + 30 -20;
power_y_dbm = 20*log10(abs(y)) + 30 -20;
power_y_kernel_dbm = 20*log10(abs(y_kernel)) + 30 -20;

%% Normalização

% Encontre os valores máximo e mínimo da potência de entrada
min_entrada = min(power_x_dbm);
max_entrada = max(power_x_dbm);

% Encontre os valores máximo e mínimo da potência de saída
min_saida = min(power_y_dbm(1:6572));
max_saida = max(power_y_dbm);

% Encontre os valores máximo e mínimo da potência de saída
min_saida_kernel = min(power_y_kernel_dbm(1:6572));
max_saida_kernel = max(power_y_kernel_dbm);

% Normalize a potência de entrada para o intervalo [0, 1]
power_x_norm = (power_x_dbm - min_entrada)/(max_entrada - min_entrada);

% Normalize a potência de saída para o intervalo [0, 1]
power_y_norm = (power_y_dbm - min_saida) / (max_saida - min_saida);

% Normalize a potência de saída para o intervalo [0, 1]
power_y_kernel_norm = (power_y_kernel_dbm - min_saida_kernel) / (max_saida_kernel - min_saida_kernel);

%% AM/AM

figure
scatter(power_x_norm, power_y_norm, 10, 'filled');
hold on
scatter(power_x_norm, power_y_kernel_norm, 10, 'filled');
legend('s/ DPD','EX-QKRLS')
xlabel('Potência de Entrada');
ylabel('Potência de Saída');
grid

%% Ganho

figure
scatter(power_x_norm, power_y_dbm - power_x_dbm, 10, 'filled');
hold on
scatter(power_x_norm, power_y_kernel_dbm - power_x_dbm, 10, 'filled');
legend('S/ DPD','EX-QKRLS','Interpreter','Latex')
xlabel('Pot\^{e}ncia de Entrada','Interpreter','Latex');
ylabel('Ganho','Interpreter','Latex');
grid
axis([0 1 -5 5]);

%% AM/PM