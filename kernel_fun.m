function y = kernel_fun(X1,X2,ker_type,ker_param)

N1 = size(X1,2);
N2 = size(X2,2);

if strcmp(ker_type,'Gauss')
    if N1 == N2
        y = (exp(-sum((X1-X2).^2)/2*ker_param))';
    elseif N1 == 1
        y = (exp(-sum((X1*ones(1,N2)-X2).^2)/2*ker_param))';
    elseif N2 == 1
        y = (exp(-sum((X1-X2*ones(1,N1)).^2)/2*ker_param))';
    else
        warning('error dimension--')
    end
    return
end

if strcmp(ker_type, 'Gauss_complex')
    if N1 == N2
        y = (exp(-sum((X1-conj(X2)).^2)/2*ker_param)).';
    elseif N1 == 1
        y = (exp(-sum((X1*ones(1,N2)-conj(X2)).^2)/2*ker_param)).';
    elseif N2 == 1
        y = (exp(-sum((X1-conj(X2*ones(1,N1))).^2)/2*ker_param)).';
    else
        warning('error dimension--')
    end
    return
end

if strcmp(ker_type, 'Gauss_complex2')
    if N1 == N2
        y = (exp(-sum(abs((X1-X2)).^2)/2*ker_param)).';
    elseif N1 == 1
        y = (exp(-sum(abs((X1*ones(1,N2)-X2)).^2)/2*ker_param)).';
    elseif N2 == 1
        y = (exp(-sum(abs((X1-X2*ones(1,N1))).^2)/2*ker_param)).';
    else
        warning('error dimension--')
    end
    return
end

if strcmp(ker_type,'Poly')
    if N1 == N2
        y = ((1 + sum(X1.*X2)).^ker_param)';
    elseif N1 == 1
        y = ((1 + X1'*X2).^ker_param)';
    elseif N2 == 1
        y = ((1 + X2'*X1).^ker_param)';
    else
        warning('error dimension--')
    end
    return

end

if strcmp(ker_type,'Cosine')
    if N1 == N2
        y = ((X1'*X2)/(norm(X1)*norm(X2)))';
    elseif N1 == 1
        y = (((X1*ones(1,N2))'*X2)/(norm(X1*ones(1,N1))*norm(X2)))';
    elseif N2 == 1
        y = ((X1'*(X2*ones(1,N1)))/(norm(X1)*norm(X2*ones(1,N2))))';
    else
        warning('error dimension--')
    end
    return
end

if strcmp(ker_type,'Laplacian')
    if N1 == N2
        y = (exp((norm(X1-X2))/sqrt(ker_param)))';
    elseif N1 == 1
        y = (exp((norm(X1*ones(1,N2)-X2))/sqrt(ker_param)))';
    elseif N2 == 1
        y = (exp((norm(X1-X2*ones(1,N1)))/sqrt(ker_param)))';
    else
        warning('error dimension--')
    end
    return
end

if strcmp(ker_type,'Sigmoid')
    a = 2;
    b = 1;
    if N1 == N2
        y = (tanh(a*X1'*X2 + b))';
    elseif N1 == 1
        y = (tanh(a*(X1*ones(1,N2))'*X2 + b))';
    elseif N2 == 1
        y = (tanh(a*X1'*(X2*ones(1,N1)) + b))';
    else
        warning('error dimension--')
    end
    return
end


warning('no such kernel')


    
return