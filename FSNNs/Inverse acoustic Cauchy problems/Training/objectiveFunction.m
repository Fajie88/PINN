function [loss,gradientsV] = objectiveFunction(parametersV,dljiaodu,dlbp_x_zhenshi,dlbp_y_zhenshi,dlBC,dlinp_x,dlinp_y,parameterNames,parameterSizes,dlflux)

% Convert parameters to structure of dlarray objects.
parametersV = dlarray(parametersV);
parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);
% Evaluate model gradients and loss.
accmodelGradients = dlaccelerate(@modelGradients);  % 加速
[gradients,loss] = dlfeval(accmodelGradients,parameters,dljiaodu,dlbp_x_zhenshi,dlbp_y_zhenshi,dlBC,dlinp_x,dlinp_y,dlflux);
% Return loss and gradients for fmincon.
gradientsV = parameterStructToVector(gradients);
gradientsV = extractdata(gradientsV);
loss = extractdata(loss);    
end