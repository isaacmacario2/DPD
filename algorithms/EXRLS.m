function [y, mse, w, P_R] = EXRLS(X, d, lambda, q1, q2, alpha, init, type)

[m, N] = size(X);
y = zeros(N, 1);  % Previsão da saída
mse = zeros(N,1);
w = zeros(m,1);
if type == 0 
    P_R = zeros(m,m,N);
end

P = init*eye(m);
   
for n = 1:N
    u = X(:,n);

    r = q2*lambda^n + u'*P*u;
    r_inv = 1/r;
    Kgain = alpha*P*u*r_inv;
    if type == 0
        e = d(n) - u'*w;
        w = alpha*w + Kgain*e;
    elseif type == 1
        e = d(n) - w'*u;
        w = alpha*w + Kgain*conj(e);
    end
    P = (alpha^2)*P - Kgain*Kgain'*r + (lambda^n)*q1*eye(m);

    if type == 0
        P_R(:,:,n) = P;
    end

    mse(n) = abs(e)^2;
end
          
end