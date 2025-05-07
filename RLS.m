function [y, mse, w] = RLS(X, d, lambda)

    [Nw, N] = size(X);
    y = zeros(N, 1);  % Previsão da saída
    mse = zeros(N,1);
    w = zeros(Nw,N);
    lambda_inv = 1/lambda;

    Sd = eye(Nw);

    for k=1:N
        e = d(k) - w(:,k)'*X(:,k);
        psi = Sd*X(:,k);
        Sd = lambda_inv*(Sd-(psi*psi')/(lambda+psi'*X(:,k)));
        vetk = Sd*X(:,k);
        w(:,k+1) = w(:,k) + conj(e)*vetk;
        mse(k) = abs(e)^2;
    end 

    w = w(:,end);

end

% function [y, mse, w] = RLS(X, d, lambda)
%     [Nw, N] = size(X);
%     y = zeros(N, 1);      % Previsão da saída
%     mse = zeros(N, 1);    % Erro quadrático médio
%     w = zeros(Nw, N);     % Matriz para armazenar pesos ao longo do tempo
%     lambda_inv = 1 / lambda;
%     P = eye(Nw);          % Matriz de covariância inicial
% 
%     for i = 1:N
%         if i == 1
%             wi = zeros(Nw,1);  % Inicialização
%         else
%             wi = w(:,i-1);     % Peso anterior
%         end
% 
%         xi = X(:,i);
%         ei = d(i) - wi' * xi;
%         Pi = lambda_inv * (P - (P * xi * xi' * P) / (lambda + xi' * P * xi));
%         ki = Pi * xi;          % Vetor ganho
%         wi = wi + ki * xonj(ei);     % Atualização dos pesos
% 
%         P = Pi;
%         w(:,i) = wi;
%         y(i) = wi' * xi;
%         mse(i) = abs(ei)^2;
%     end
% 
%     w = w(:,end);  % Retorna os últimos pesos
% end