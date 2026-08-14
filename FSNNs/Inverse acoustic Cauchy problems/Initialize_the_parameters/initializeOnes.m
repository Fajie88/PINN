function parameter = initializeOnes(sz,className)
arguments
    sz
    className = 'single'
end
parameter = 0.1.*ones(sz,className);
parameter = dlarray(parameter);
end