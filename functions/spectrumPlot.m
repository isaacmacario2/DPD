function spectrumPlot(N,fs,x1,fun_type1,x2,fun_type2)

% Verificar se o segundo sinal foi fornecido 
if nargin < 6 
    x2 = []; 
    fun_type2 = ''; 
end

if strcmp(fun_type1,'fft')
    X1 = fftshift(fft(x1, N));
elseif strcmp(fun_type1,'pwelch')
    X1 = fftshift(pwelch(x1, N));
end

% Calcular a potência do sinal
Pot1 = abs(X1).^2/(N);

% Converter para dBm
P_signal_dBm1 = 10*log10(Pot1) + 30;

% Vetor de frequências
f = (-fs/2:fs/N:fs/2-fs/N);

% Plotar o espectro
figure;
plot(f/1e6, P_signal_dBm1, 'LineWidth', 1.5);
hold on;

% Verificar e processar o segundo sinal, se fornecido 
if ~isempty(x2) 
    if strcmp(fun_type2, 'fft') 
        X2 = fftshift(fft(x2, N)); 
    elseif strcmp(fun_type2, 'pwelch') 
        X2 = fftshift(pwelch(x2, N)); 
    end
     
    Pot2 = abs(X2).^2 / N; 
    P_signal_dBm2 = 10 * log10(Pot2) + 30; 
    plot(f/1e6, P_signal_dBm2, 'LineWidth', 1.5); 
end

xlabel('Frequência (MHz)');
ylabel('Potência (dBm)');
title('Espectro do Sinal em dBm');
grid on;
xlim([-fs/2 fs/2]/1e6);

if ~isempty(x2) 
    legend('Sinal 1', 'Sinal 2'); 
else 
    legend('Sinal 1'); 
end

end
