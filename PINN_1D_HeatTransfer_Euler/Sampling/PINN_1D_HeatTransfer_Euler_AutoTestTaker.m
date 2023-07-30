%{
A MATLAB program to calculate 1D Heat Transfer problem. Neural Network
is assigned to solve the Partial Differential Equations in problem.
1D Heat Transfer problem is described as "dy/dt = alpha*(d2y/dx2)".

Created by:
Theodoret Putra Agatho
University of Atma Jaya Yogyakarta
Department of Informatics
09/05/2023

Disclaimer:
This program writing is influenced heavily by following code:
Andreas Almqvist (2023). Physics-informed neural network solution of 2nd order ODE:s 
(https://www.mathworks.com/matlabcentral/fileexchange/96852-physics-informed-neural-network-solution-of-2nd-order-ode-s),
MATLAB Central File Exchange. Retrieved May 11, 2023.
%}

clear all; clc; close all;
%% Initialization

% Hyperparameters
alpha = 1.0;

% Time
dt = 0.01;
time = 0.1; % time max
epoch_time = floor(time/dt);

% Grid 1D
Nx = 41; % N-many elements in x vector
xmin = 0; xmax = 1; % boundary x vector
x = linspace(xmin,xmax,Nx); % Grid
dx = x(2) - x(1);

% Neural Network
learning_rate = 0.001;
epoch_learning = 1024;
batch = 128;
Nn = 32; % N-many neuron
% weightes and biases
min = -2; max = 2; % boundary weights and biases
w0 = min + rand(Nn,1)*(abs(min)+abs(max));
b0 = min + rand(Nn,1)*(abs(min)+abs(max));
w1 = min + rand(Nn,1)*(abs(min)+abs(max));
b1 = min + rand(1)*(abs(min)+abs(max));
params = [w0;b0;w1;b1];
% initial_params = params;

% Initial values
U0 = sin(pi*x); % consider "U" as current y and "y" as next y


% Matrix list sample test initialization
test_iterations = 10;
list_tnn = zeros(test_iterations,1);
list_tfd = zeros(test_iterations,1);
list_Unn = zeros(test_iterations,Nx);
list_Ufd = zeros(test_iterations,Nx);


%% Other solutions
% Numerical solution - Finite Difference Method
s = alpha*dt/dx^2;

A = eye([Nx,Nx]);
for i = 2:Nx-1
    for j = 2:Nx-1
        if A(i,j) == 1
            A(i,j) = 1 + 2*s;
            A(i,j-1) = -s;
            A(i,j+1) = -s;
        end
    end
end

for k = 1:test_iterations
    Ufd = U0;
    t0fd = tic; % stopwatch start
    for i = 1:epoch_time
        Ufd = Ufd/A;
        Ufd(1) = 0; Ufd(end) = 0; % boundary conditions
    end
    tfd = toc(t0fd); % stopwatch end
    list_Ufd(k,:) = Ufd;
    list_tfd(k) = tfd;
end

% Exact solution
Ue = sin(pi.*x).*exp(1).^(-pi^2.*time);




%% Neural Network training
for k = 1:test_iterations
    U = U0;
    t0nn = tic; % stopwatch start
    for i = 1:epoch_time
        disp(k+"."+i);
        for j = 1:epoch_learning
            params = training(params,U,x,Nn,alpha,dt,batch,learning_rate);
        end
        [y,~,~] = predict(params,x,Nn);
        U = y;
    end
    tnn = toc(t0nn); % stopwatch end
    list_Unn(k,:) = U;
    list_tnn(k) = tnn;
end



%% Results
% Error
error_nn = mean((list_Unn-Ue).^2,'all');
error_fd = mean((list_Ufd-Ue).^2,'all');

% Time
time_nn = mean(list_tnn);
time_fd = mean(list_tfd);



%% Functions (Neural Network)
function y = mySigmoid(x)
    y = 1./(1 + exp(-x));
end

function params = training(params,U,x,Nn,alpha,dt,batch,learning_rate)
    w0 = params(1:Nn);
    b0 = params(Nn+1:Nn*2);
    w1 = params(Nn*2+1:end-1);
    b1 = params(end);
    
    for j = 1:batch
        % Pick a random data point for current batch
        i = randi(length(x));
        xi = x(i);
        Ui = U(i);

        % Sigmoid function derivatives with w0, b0, and xi inputs
        s = mySigmoid(w0*xi + b0);
        dsdx = s.*(1 - s);
        d2sdx2 = dsdx.*(1 - 2*s);
        d3sdx3 = d2sdx2.*(1 - 2*s) - 2*dsdx.^2;
        % at boundary
        s0 = mySigmoid(w0*x(1) + b0);
        s1 = mySigmoid(w0*x(end) + b0);
    
        % Prediction current batch
        y = sum(w1.*s) + b1;
        % dydx = sum(w0.*w1.*dsdx); % unused
        d2ydx2 = sum(w0.^2.*w1.*d2sdx2);
        % at boundary
        y0 = sum(w1.*s0) + b1; 
        y1 = sum(w1.*s1) + b1;

        % y derivatives to weights and biases
        dydw0 = w1.*dsdx.*xi;
        dydb0 = w1.*dsdx;
        dydw1 = s;
        dydb1 = 1;

        % dydx derivatives to weights and biases - unused
        % dypdw0 = w1.*dsdx + w0.*w1.*d2sdx2.*xi; % yp = dydx
        % dypdb0 = w0.*w1.*d2sdx2;
        % dypdw1 = w0.*dsdx;
        % dypdb1 = 0;

        % d2ydx2 derivatives to weights and biases
        dyppdw0 = 2*w0.*w1.*d2sdx2 + w0.^2.*w1.*d3sdx3.*xi; % ypp = d2ydx2
        dyppdb0 = w0.^2.*w1.*d3sdx3;
        dyppdw1 = w0.^2.*d2sdx2;
        dyppdb1 = 0;

        ds0dx = s0.*(1 - s0);
        % y0 derivatives to weights and biases
        dy0dw0 = w1.*ds0dx.*x(1);
        dy0db0 = w1.*ds0dx;
        dy0dw1 = s0;
        dy0db1 = 1;

        ds1dx = s1.*(1 - s1);
        % y1 derivatives to weights and biases
        dy1dw0 = w1.*ds1dx.*x(end);
        dy1db0 = w1.*ds1dx;
        dy1dw1 = s1;
        dy1db1 = 1;

        % Weights and biases update
        % l = mean((y - dt*alpha*d2ydx2 - Ui).^2) + (y0)^2 + (y1)^2; % loss function
        w0 = w0 - learning_rate*(2*(y - dt*alpha*d2ydx2 - Ui)* ...
            (dydw0 - dt*alpha*dyppdw0) + 2*y0*dy0dw0 + 2*y1*dy1dw0);
        b0 = b0 - learning_rate*(2*(y - dt*alpha*d2ydx2 - Ui)* ...
            (dydb0 - dt*alpha*dyppdb0) + 2*y0*dy0db0 + 2*y1*dy1db0);
        w1 = w1 - learning_rate*(2*(y - dt*alpha*d2ydx2 - Ui)* ...
            (dydw1 - dt*alpha*dyppdw1) + 2*y0*dy0dw1 + 2*y1*dy1dw1);
        b1 = b1 - learning_rate*(2*(y - dt*alpha*d2ydx2 - Ui)* ...
            (dydb1 - dt*alpha*dyppdb1) + 2*y0*dy0db1 + 2*y1*dy1db1);
    end
    params = [w0;b0;w1;b1];
end

function [y,dydx,d2ydx2] = predict(params,x,Nn)
    w0 = params(1:Nn);
    b0 = params(Nn+1:Nn*2);
    w1 = params(Nn*2+1:end-1);
    b1 = params(end);

    Nx = length(x);
    w0 = repmat(w0,1,Nx);
    b0 = repmat(b0,1,Nx);
    w1 = repmat(w1,1,Nx);
    x = repmat(x,Nn,1);

    % Sigmoid function derivative with w0, b0, and x inputs
    s = mySigmoid(w0.*x + b0);
    dsdx = s.*(1 - s);
    d2sdx2 = dsdx.*(1 - 2*s);

    % Prediction
    y = sum(w1.*s) + b1;
    dydx = sum(w0.*w1.*dsdx);
    d2ydx2 = sum(w0.^2.*w1.*d2sdx2);
end