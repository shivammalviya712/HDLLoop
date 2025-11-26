function result = calculate_factorial(n)
    % Calculate the factorial of a number
    % Input: n (integer)
    % Output: result (integer)
    
    if n == 0
        result = 1;
    else
        result = 1;
        for i = 1:n
            result = result * i;
        end
    end
    
    disp(['Factorial of ', num2str(n), ' is ', num2str(result)]);
end

%% Visualization Section
x = 0:0.1:10;
y = sin(x);
plot(x, y);
title('Sine Wave');
grid on;

