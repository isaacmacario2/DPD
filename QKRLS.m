function [a,D,y,mse,u] = QKRLS(X,d,IN,lambda,sigma_k,delta,Lmax)
 
    [~, N] = size(X);
    %X = X';
    D = [];                          % Dicionário de vetores
    D(:,1) = X(:,1);
    a = zeros(size(D,2),1);          % Vetor de coefifientes
    R = eye(size(D,2)) * 0.1;        % Matriz de auto-correlação
    y = zeros(N, 1);                 % Vetor estimado
    u = zeros(N, 1);
    beta_0 = zeros(size(D,2), 1);    % Vetor de correlação cruzada
    r = zeros(size(D,2), 1);         % Vetor de resíduos
    mse = zeros(N,1);
    H = 1;                        % Amplitude inicial de busca
    Mb = 16;                      % Número máximo de bits de atualização
    Nu = 8;                       % Número máximo de iterações bem-sucedidas

    for k = 2:N
    
        h = zeros(size(a,1), 1); % Inicializa h_n com zeros
        h_in = zeros(size(a,1), 1); % Inicializa h_n com zeros
        for l = 1:size(D, 2)
            h(l) = kernel_fun(X(:,k), D(:,l),'Gauss_complex', sigma_k).';
            h_in(l) = kernel_fun(IN(:,k), D(:,l),'Gauss_complex', sigma_k).';
            y(k) = y(k) + h(l)'*a(l) + h(l)*a(l);
            u(k) = u(k) + h_in(l)*a(l) ; %+ h_in(l)'*a(l)
        end
          
          % u(k) = h_in.'*a;
        
        if isempty(D)
            dist_min = inf;
        else
            [dist_min, idx_min] = min(arrayfun(@(l) norm(X(:, k) - D(:, l)), 1:size(D, 2)));
        end

        if dist_min >= delta
            D = [D X(:,k)];
            h = [h; 1]; 
            R = lambda * blkdiag(R, 1) + (h*h'); % Atualiza matriz de correlação

            e = d(k) - h.'*[a; 0];

            beta_0 = lambda*[r; 0] + e*h;

            [delta_a, r] = DCD(R, beta_0, H, Mb, Nu);

            a = [a; 0] + delta_a;
            
            if size(D, 2) > Lmax
                Ek = zeros(1, size(D, 2));
                for l = 1:size(D, 2)
                    Ek(l) = abs(a(l)) * kernel_fun(X(:, k), D(:, l), 'Gauss_complex', sigma_k);
                end
                [~, idx_remover] = min(Ek);
            
                D(:, idx_remover) = [];
                R(idx_remover, :) = [];
                R(:, idx_remover) = [];
                a(idx_remover) = [];
                r(idx_remover) = [];
            end
        else
            
            e = d(k) - h.'*a;
            a(idx_min) = a(idx_min) + e;
        end
    
        mse(k) = abs(e)^2;
    end

end