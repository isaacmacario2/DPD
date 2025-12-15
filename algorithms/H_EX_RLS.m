function [y, mse, w, P_H] = H_EX_RLS(X, d, lambda, q1, q2, alpha, init, flag, type)

[m, N] = size(X);
y = zeros(N, 1);  % Previsão da saída
mse = zeros(N,1);
w = zeros(m,1);
lambda_inv = 1/lambda;
if type == 0
    P_H = zeros(m,m,N);
end

S_inv = sqrt(init)*eye(m);

for k = 1:N
    u = X(:,k);

    c = (S_inv'*u)/((lambda^(1/2))*sqrt(q2));

    delta = sqrt(1+(c'*c));

    g = (S_inv*c)/(sqrt(q2)*(lambda^(1/2))*delta);

    S_est_inv = sqrt(lambda_inv)*S_inv-(sqrt(q2)/(delta+1))*g*c';

    %%%%%%%%%%%%%%%%%%
    if flag == 0
        Ps = S_est_inv * S_est_inv';
        Ps = (alpha^2)*Ps + q1*eye(m);
        S_inv = chol(Ps, 'lower');
    end
    %%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if flag == 1
        S_inv = alpha*S_est_inv;
        for i = 1:m
            S_inv(i,i) = sqrt(S_inv(i,i)^2 + q1);
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if flag == 2
        S_inv = alpha*S_est_inv;
        for i = 1:m
            S_inv(i,i) = S_inv(i,i) + q1/(2*S_inv(i,i));
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if type == 0
        e = d(k) - u'*w;
        w = alpha*w + alpha*(e/delta)*g;
        P_H(:,:,k) = S_inv;
    elseif type == 1
        e = d(k) - w'*u;
        w = alpha*w + alpha*(conj(e)/delta)*g;
    end


    mse(k) = abs(e)^2;
          
end

end