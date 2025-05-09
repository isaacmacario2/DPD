function [a, dic, y, mse] = QKLMS(X, d, mu, sigma_k, delta, Lmax)
 
    [N, ~] = size(X);
    a = [];           % Coeficientes
    dic = [];         % Centros dos kernels
    y = zeros(N, 1);  % Previsão da saída
    mse = zeros(N,1);

    for k = 1:N
        if isempty(dic)
            y(k) = 0;
        else
            for l = 1:length(a)
                h(l) = kernel_fun(X(k, :), dic(l, :),'Gauss_complex', sigma_k);
                y(k) = y(k) + h(l)*a(l);
            end
        end

        % Erro a priori
        e_n = d(k) - y(k);

        % MSE
        mse(k) = abs(e_n).^2;

        if isempty(dic)
            dist_min = inf;
        else
            [dist_min, idx_min] = min(arrayfun(@(l) norm(X(k, :) - dic(l, :)), 1:size(dic, 1)));
        end

        if dist_min >= delta
            % Adicionar novo centro e atualizar coeficiente
            a = [a; mu * e_n];
            dic = [dic; X(k, :)];

            if size(dic, 2) > Lmax
                Ek = zeros(1, size(dic, 2));
                for l = 1:size(dic, 2)
                    Ek(l) = abs(a(l)) * kernel_fun(X(:, k), dic(:, l), 'Gauss_complex', sigma_k);
                end
                [~, idx_remover] = min(Ek);
            
                dic(:, idx_remover) = [];
                a(idx_remover) = [];
            end
        else
            % Atualizar o coeficiente do centro mais próximo
            a(idx_min) = a(idx_min) + mu * e_n;
        end
    end
end
