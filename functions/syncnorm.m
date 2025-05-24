function y_sync = syncnorm(x,y,flag1,flag2,flag3)

    [c, lag] = xcorr(y, x);
    [~, I] = max(abs(c));
    lagDiff = lag(I);
    if lagDiff > 0
        y_sync = [y(lagDiff+1:end)];
        % y_sync = [y(7:end)];
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
    
    %  % Normalização
    % if flag1 == 1
    %     y_sync = y_sync * norm(x) / norm(y_sync);
    % elseif flag1 == 2
    %     y_sync = y_sync / norm(y_sync);
    % elseif flag1 == 3
    %     PeakReal=max(abs(real(y_sync)));  
    %     PeakImag=max(abs(imag(y_sync)));
    %     Peak=max([PeakReal PeakImag]);
    %     if Peak ~=0
    %         y_sync=double(y_sync./Peak);
    %     else
    %         y_sync=double(y_sync);
    %     end
    % end  


        % Atraso de subsample
    if flag2 == 1
        % D = [[y_sync(2:end); 0] [0; y_sync(1:end-1)]];
        D = [y_sync [0; y_sync(1:end-1)]];
        coeffs = (D'*D) \ (D'*x);
        y_sync = D*coeffs;
    end


    if flag3==1
        y_sync = ifft(fft(y_sync).*exp(-phdiffmeasure(x, y_sync)*1i));
    end

   % Normalização
    if flag1 == 1
        y_sync = y_sync * norm(x) / norm(y_sync);
    elseif flag1 == 2
        y_sync = y_sync / norm(y_sync);
    elseif flag1 == 3
        PeakReal=max(abs(real(y_sync)));  
        PeakImag=max(abs(imag(y_sync)));
        Peak=max([PeakReal PeakImag]);
        if Peak ~=0
            y_sync=double(y_sync./Peak);
        else
            y_sync=double(y_sync);
        end
    end  


    
end 