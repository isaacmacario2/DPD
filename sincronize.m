function y_sync = sincronize(x,y,flag1,flag2)

    [c, lag] = xcorr(y, x);
    [~, I] = max(abs(c));
    lagDiff = lag(I);
    if lagDiff > 0
        y_sync = [y(lagDiff+1:end)];
        length_input = length(x);
        length_output = length(y_sync);
        y_sync = [y_sync; zeros(length_input - length_output, 1)];
    elseif lagDiff < 0
        y_sync = y(1:end+lagDiff);
        length_input = length(x);
        length_output = length(y_sync);
        y_sync = [zeros(length_input - length_output, 1); y_sync];
    else
        y_sync = y;
    end
    
    % Atraso de subsample
    if flag2 == 1
        D = [[y_sync(2:end); 0] [0; y_sync(1:end-1)]];
        % D = [y_sync [0; y_sync(1:end-1)]];
        coeffs = (D'*D) \ (D'*x);
        y_sync = D*coeffs;
    end

% Normalização
    if flag1 == 1
        y_sync = y_sync * norm(x) / norm(y_sync);
    end

    if flag1 == 2
        y_sync = y_sync / norm(y_sync);
    end
    
end 