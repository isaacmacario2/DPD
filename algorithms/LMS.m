function [y, mse, w] = LMS(X, d, mu)
    
    [Nw, N] = size(X);
    y = zeros(N, 1);  % Previsão da saída
    mse = zeros(N,1);
    w = zeros(Nw,N);

    for k = 1:N      
        % Computando erro a priori:
        e = d(k) - w(:,k)'*X(:,k);
        w(:,k) = w(:,k) + mu*conj(e)*X(:,k);
        mse(k) = abs(e)^2;
        y(k) = w(:,k)'*X(:,k);
    end % for k 

    w = w(:,end);

end
