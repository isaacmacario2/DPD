function [w, mse, u, y] = EX_RFF_RLS(X, d, IN, sigma_k, D, alpha, beta, q1, q2, flag,tipo)

% --- Inicializações
[p, N] = size(X);
y = zeros(N,1);
u = zeros(N,1);
mse = zeros(N,1);
beta_inv = 1/beta;

if flag == 0 
    % --- Geração das Random Fourier Features
    Omega = randn(p, D) / sigma_k;      % frequências aleatórias ~ N(0, 1/sigma_k^2)
    theta = 2*pi*rand(D, 1);            % fase aleatória ~ U(0, 2pi)
    
    % z = sqrt(2/D) * cos(Omega'*X(:,1) + theta); 
    
    % --- Inicialização dos parâmetros do EX-RLS
    w = zeros(D,1);                 % vetor de coeficientes
    P = 100*eye(D);

    % --- Loop principal
    for n = 1:N 
        x = X(:,n);
        z = sqrt(2/D) * cos(Omega'*x + theta);
    
        % Ganho de atualização
         r = q2*(beta^n) + z'*P*z;
         r_inv = 1/r;
         Kgain = alpha*P*z*r_inv;
         % e = d(n) - w'*z;
         e = d(n) - z'*w;
         P = alpha^2*P - Kgain*Kgain'*r + (beta^n)*q1*eye(D);
         % w = w + Kgain*conj(e);
         w = w + Kgain*e;


        % MSE
        mse(n) = abs(e)^2;
    end
elseif flag == 1
    Xr = [real(X); imag(X)];
    INr = [real(IN); imag(IN)];
    % Xr = X;

    % --- Geração das Random Fourier Features
    Omega = randn(2*p, D) / sigma_k;  % ω ~ N(0, 1/σ^2 I)
    % Omega = (randn(p, D) + 1j*randn(p, D)) / (sqrt(2)*sigma_k);
    theta = 2*pi*rand(D,1);           % θ ~ U(0, 2π)
    
    % z = sqrt(2/D) * cos(Omega'*X(:,1) + theta); % Z: D x N
    
    % --- Inicialização dos parâmetros do EX-RLS
    w = zeros(D,1);                 % vetor de coeficientes
    P = 100*eye(D);
    S_inv = 100*eye(D);
    
    % --- Loop principal
    for n = 1:N 
        if mod(n,100) == 0
            disp(n)
        end
        x = Xr(:,n);
        zC = sqrt(1/D)*exp(1j*(Omega'*x + theta));
        y(n) = w'*zC;

        if ~isempty(IN)          
            in = INr(:,n);
            zC_in = sqrt(1/D)*exp(1j*(Omega'*in + theta));
            u(n) = w'*zC_in;
        end
       
        if tipo == 1
            % Ganho de atualização
            r = q2*(beta^n) + zC'*P*zC;
            r_inv = 1/r;
            Kgain = alpha*P*zC*r_inv;
            e = d(n) - w'*zC;
            % e = d(n) - zC'*w;
            P = alpha^2*P - Kgain*Kgain'*r + (beta^n)*q1*eye(D);
            w = w + Kgain*conj(e);
            % w = w + Kgain*e;
        elseif tipo == 2
            c = (S_inv'*zC)/((beta^(1/2))*sqrt(q2));
            delta = sqrt(1+(c'*c));
            g = (S_inv*c)/(sqrt(q2)*(beta^(1/2))*delta);
            S_est_inv = sqrt(beta_inv)*S_inv-(sqrt(q2)/(delta+1))*g*c';
            Ps = S_est_inv * S_est_inv';
            Ps = (alpha^2)*Ps + q1*eye(D);
            S_inv = chol(Ps, 'lower');
            e = d(n) - w'*zC;
            % e = d(n) - zC'*w;
            w = alpha*w + alpha*(conj(e)/delta)*g;
            % w = alpha*w + alpha*(e/delta)*g;
        elseif tipo == 3
            ak = (1/(sqrt(q2*beta)))*S_inv'*zC;
            igamma = 1;
            ctheta = zeros(1,D);
            stheta = zeros(1,D);
            for k = 1:D %m:-1:1 %
                aux1      = sqrt(abs(igamma)^2+abs(ak(k))^2);
                ctheta(k) = abs(igamma)/aux1; %
                stheta(k) = -(-ak(k)/aux1);
                igamma    = aux1;
            end
            gamma = 1/(igamma);

            sHaux = zeros(D,1);
            SmHaux = (1/sqrt(beta*q2))*S_inv';
            for k = 1:D
                for i = 1:D
                    aux2 = sHaux(i);
                    sHaux(i) = ctheta(k)*aux2 - conj(stheta(k))*SmHaux(k,i);
                    SmHaux(k,i) = (stheta(k)*aux2 + ctheta(k)*SmHaux(k,i))*sqrt(q2);
                end
            end          
            s = conj(sHaux);%/sqrt(q2);         
            S_est_inv = SmHaux';
            Ps = S_est_inv * S_est_inv';
            Ps = (alpha^2)*Ps + q1*eye(D);
            S_inv = chol(Ps, 'lower');
            e = d(n) - w'*zC;
            w = alpha*w - alpha*(conj(e)*gamma)*s;     
        end
    
        % MSE
        mse(n) = abs(e)^2;
    end
    % u = [];
% if ~isempty(IN)
%     INr = [real(IN); imag(IN)];
%     u = zeros(1, size(INr,2));
%     for k = 1:size(INr,2)
%         in = INr(:,k);
%         zC_in = sqrt(1/D)*exp(1j*(Omega'*in + theta));
%         u(k) = w' * zC_in;
%     end
% end
end

return
