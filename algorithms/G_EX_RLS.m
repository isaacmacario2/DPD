function [y, mse, w, P_G] = G_EX_RLS(X, d, lambda, q1, q2, alpha, init, flag, type)

[m, N] = size(X);
y = zeros(N, 1);  % Previsão da saída
mse = zeros(N,1);
w = zeros(m,1);
if type == 0
    P_G = zeros(m,m,N);
end

S_inv = sqrt(init)*eye(m);

for k = 1:N
    u = X(:,k);

    %--------------------------------------
    % Obtaining a(k)
    %--------------------------------------
    ak = (1/(sqrt(q2*lambda)))*S_inv'*u;


    %----------------------------------
    % Obtaining Qtheta(k) and gamma(k):
    %----------------------------------
    igamma = 1;
    ctheta = zeros(1,m);
    stheta = zeros(1,m);
    for n = 1:m %m:-1:1 %
        aux1      = sqrt(abs(igamma)^2+abs(ak(n))^2);
        ctheta(n) = abs(igamma)/aux1; %
        % stheta(n) = -(-ak(n)/igamma)*ctheta(n);
        stheta(n) = -(-ak(n)/aux1);
        igamma    = aux1;
    end
    gamma = 1/(igamma);


    %---------------------------------------
    % Obtaining s(k) and updating S^{H}(k):
    %---------------------------------------
    sHaux = zeros(m,1);
    SmHaux = (1/sqrt(lambda*q2))*S_inv';
    for n = 1:m
        for i = 1:m
            aux2 = sHaux(i);
            sHaux(i) = ctheta(n)*aux2 - conj(stheta(n))*SmHaux(n,i);
            SmHaux(n,i) = (stheta(n)*aux2 + ctheta(n)*SmHaux(n,i))*sqrt(q2);
        end
    end
    s = conj(sHaux);%/sqrt(q2);
    S_est_inv = SmHaux';
  

    %---------------------------------------
    % Cholesky
    %---------------------------------------
    if flag == 0
        Ps = S_est_inv*S_est_inv';
        Ps = (alpha^2)*Ps + q1*eye(m);
        S_inv = chol(Ps,'lower');
    end

    %---------------------------------------
    % Aproximação por raíz quadrada
    %---------------------------------------
    if flag == 1
        S_inv = alpha*S_est_inv;
        for i = 1:m
            S_inv(i,i) = sqrt(S_inv(i,i)^2 + q1);
        end
    end

    %---------------------------------------
    % Aproximação por Taylor
    %---------------------------------------
    if flag == 2 
        S_inv = alpha*S_est_inv;
        for i = 1:m
            S_inv(i,i) = S_inv(i,i) + q1/(2*S_inv(i,i));
        end
    end

    if type == 0
        e = d(k) - u'*w;
        w = alpha*(w - e*(gamma)*s);
        P_G(:,:,k) = S_inv;
    elseif type == 1
        e = d(k) - w'*u;
        w = alpha*(w - conj(e)*(gamma)*s);
    end

    mse(k) = abs(e)^2;

end
end