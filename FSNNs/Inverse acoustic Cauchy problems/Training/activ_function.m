%%%各类激活函数
function sig = activ_function(x)
global activation_function
% ------------------------------------------------------------
if activation_function == 1
    sig = logsig(x);                 % nonlinear function used in neural networks
elseif activation_function == 2
    sig = tanh(x); 
elseif activation_function == 3
    sig = x.*logsig(x);  
elseif activation_function == 4
    sig =  log(1 + exp(x));
elseif activation_function == 5
    sig = sin(x); 
elseif activation_function == 6 
    sig = atan(x); 
elseif activation_function == 7
    sig = x.*tanh(log(1 + exp(x))); 
end
end