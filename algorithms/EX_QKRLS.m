function [y,a,D,mse,u] = EX_QKRLS(X,d,IN,sigma_k,alpha,lambda,beta,q,delta,Lmax)

[~,N] = size(X);
mse = zeros(N,1);

D = [];
D(:,1) = X(:,1);
a = zeros(size(D,2),1);          % Vetor de coefifientes
y = zeros(N, 1);                 % Vetor estimado
u = zeros(N, 1);
x = zeros(N, 1);

s = X(1,:);

Q = 0;
ro_inv = 1/(beta*lambda);

for k = 2:N

    h = zeros(size(a,1), 1);
    h_in = zeros(size(a,1), 1); % Inicializa h_n com zeros
    for l = 1:size(D, 2)
        h(l) = kernel_fun(X(:,k), D(:,l),'Gauss_complex2', sigma_k).';
        h_in(l) = kernel_fun(IN(:,k), D(:,l),'Gauss_complex2', sigma_k).';
        y(k) = y(k) + h(l)'*a(l);%; + h(l)*a(l)
        u(k) = u(k) + h_in(l)*a(l) ; %+ h_in(l)'*a(l)
    end

    % u(k) = y(k) - s(k);% - y(k);
    % x(k) = rfpa(u(k)); 
    % d(k) = x(k);

    if isempty(D)
        dist_min = inf;
    else
        [dist_min, idx_min] = min(arrayfun(@(l) norm(X(:, k) - D(:, l)), 1:size(D, 2)));
    end

    % dist_min
    if dist_min >= delta
        D = [D X(:,k)];
        % h = [h; 1];
        z = Q*h;
        r_ro = (beta^(k) + ro_inv*kernel_fun(X(:,k),X(:,k),'Gauss_complex2',sigma_k) - h'*z);
        r_inv = 1/r_ro;

        % e = d(k) - h'*[a; 0];%
        e = d(k) - h'*a;%

        a = alpha*[(a - z*r_inv*e); alpha*ro_inv*r_inv*e];
        
        Q = (abs(alpha)^2)*[Q + z*z'*r_inv, -ro_inv*z*r_inv; (-ro_inv*z*r_inv)',ro_inv^2*r_inv];

        ro_inv = (abs(alpha)^2)*ro_inv + beta^(k)*q;

        if size(D, 2) > Lmax
            Ek = zeros(1, size(D, 2));
            for l = 1:size(D, 2)
                Ek(l) = abs(a(l)) * kernel_fun(X(:, k), D(:, l), 'Gauss_complex2', sigma_k);
            end
            [~, idx_remover] = min(Ek);

            D(:, idx_remover) = [];
            a(idx_remover) = [];
            Q(idx_remover, :) = [];
            Q(:, idx_remover) = [];
        end
    else
        e = d(k) - h'*a;
        a(idx_min) = a(idx_min) + e;
    end
    mse(k) = abs(e)^2;

end

return
