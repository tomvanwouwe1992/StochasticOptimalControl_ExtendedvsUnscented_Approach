import casadi.*
import org.opensim.modeling.*
clear all; close all; clc;
figure(1)
cs = linspecer(15);
dt = 0.01;
noiseVariances = [1 2 5 10 20 50 100 200 500 1000 2000 5000 10000];
load(['EKFClipped_Variance' num2str(1) '.mat']);
IG = pendulumResult_UKF;
for i = 2:length(noiseVariances)
    
    noiseVariance = noiseVariances(i); %N²s
    noiseVarianceDiscrete = noiseVariance/dt;
    noiseStdDiscrete = sqrt(noiseVarianceDiscrete);
       
        load(['EKFClipped_Variance' num2str(noiseVariances(i-1)) '.mat']);
        IG = pendulumResult_UKF;    L_scaling = 1e-2;
    %% Add folders with helper functions to path
    local_path = pwd;
    idcs   = strfind(local_path,'\');
    folder_path = local_path(1:idcs(end)-1);
    addpath(genpath([folder_path '/0. SharedFunctions']))
    
    
    %% UKF paramaters
    % Formulation based on kalman filtering and neural nets by simon haykin
    
    % Here there is a difference between the weights used to calculate the mean
    % from the tranformed points and those used to compose the covariance matrix after
    % transformation of the sigma points.
    
    % Note that the mean selection weights (W_M) always add up to 1!
    % The recombination of transformed points --> W_C do not need to add up to 1!
    
    nStates = 2;
    nNoiseSources = 1;
    nTOT = nStates + nNoiseSources;
    % nTOT = 0;
    
    alpha = 1; % --> determines the spread of the sigma points -- influence on c! a small value means a small spread around the mean in sigma selection. Has an influence on selection of the posterior covariance
    kappa = 0; % Scales the spread of the sigma points as well, often 3-n is chosen, but this can generate some problems with positive definitness
    beta = 2; % Is a parameter to incorporate prior knowledge on distribution, only affects the weights to compose the posterior covariance matrix
    % If alpha = 1 and kappa = 0 you get the basic implementation of the UKF.
    
    % Note that Julier commented in the paper (unscented ...) that non-positive
    % weights for predicted covariance can lead to non-positive definitness
    % kappa = -1;
    lambda = alpha^2*(nTOT + kappa) - nTOT;
    
    W_0_M = lambda/(nTOT+lambda);
    W_i_M = 1/(2*(nTOT+lambda));
    W_0_C = lambda/(nTOT+lambda) + 1 - alpha^2 + beta;
    W_i_C = 1/(2*(nTOT+lambda));
    
    WeightVec_M = [W_0_M W_i_M*ones(1,2*(nTOT))];
    WeightVec_C = [W_0_C W_i_C*ones(1,2*(nTOT))];
    c = sqrt(nTOT+lambda);
    
    
    %- Stochastic Dynamics of the controlled system
    % Variables initialisaiton
    X = MX.sym('X',1,1);
    dX = MX.sym('dX',1,1);
    F_dist =  MX.sym('F_dist',1,1);
    F_1 =  MX.sym('F_1',1,1);
    F_2 =  MX.sym('F_2',1,1);
    
    K_1 = MX.sym('K_1',1,1);
    B_1 = MX.sym('B_1',1,1);
    K_2 = MX.sym('K_2',1,1);
    B_2 = MX.sym('B_2',1,1);
    % Dynamic System parameters
    g = 9.81; l = 1.0; m = 75;
    
    % Dynamics of stochastic problem ( xdot = f(x,u,...) )
    sharpnessFactor = 50;
    lowerClippingValue = 0;
    upperClippingValue = 250;
    
    F_control_1 = F_1 + K_1*(X-pi) + B_1*dX;
    F_control_1_clipped = smoothClipping_tanh(F_control_1,sharpnessFactor,lowerClippingValue,upperClippingValue);
    
    F_control_2 = F_2 + K_2*(X-pi) + B_2*dX;
    %         F_control_2 = -F_control_2;
    F_control_2_clipped = -smoothClipping_tanh(F_control_2,sharpnessFactor,lowerClippingValue,upperClippingValue);
    
    
    fcn_smoothClipping_tanh = Function('fcn_smoothClipping_tanh',{F_1,X,dX,K_1,B_1},{F_control_1_clipped});
    
    derivative_smoothClipping_tanh = jacobian(F_control_1_clipped,[X dX]);
    fcn_derivative_smoothClipping_tanh = Function('fcn_derivative_smoothClipping_tanh',{F_1,X,dX,K_1,B_1},{derivative_smoothClipping_tanh});
    
    
    x_test = [lowerClippingValue - 10:0.01:upperClippingValue + 10];
    x_testClipped = smoothClipping_tanh(x_test,sharpnessFactor,lowerClippingValue,upperClippingValue);
    plot(x_test,x_testClipped);
    
    Xdot = [ dX; (-m*g*l*sin(X) + F_control_1_clipped + F_control_2_clipped + F_dist)/(m*l^2)];
    fcn_OL = Function('fcn_OL',{dX,X,F_1,F_2,F_dist,K_1,B_1,K_2,B_2},{Xdot});
    
    
    A = jacobian(Xdot,[X dX]);
    A_fcn = Function('A_fcn',{dX,X,F_1,F_2,F_dist,K_1,B_1,K_2,B_2},{A});
    
    C = jacobian(Xdot,F_dist);
    C_fcn = Function('C_fcn',{dX,X,F_1,F_2,F_dist,K_1,B_1,K_2,B_2},{C});
    
    
    clear X dX F K B
    
    %- Variables and controls
    opti = casadi.Opti();
    
    
    %% States/Controls of the nominal system
    % X_nom = opti.variable(1,(N+1));
    % dX_nom = opti.variable(1,(N+1));
    X = opti.variable(1,1);
    dX = opti.variable(1,1);
    F_1 = opti.variable(1,1);
    F_2 = opti.variable(1,1);
    
    opti.subject_to(F_1 > 1e-2);
    opti.subject_to(F_2 > 1e-2);
    
    L = [L_scaling;L_scaling;L_scaling].*opti.variable(3,1);
    K_1 = 1000*opti.variable(1,1);
    B_1 = 100*opti.variable(1,1);
    K_2 = 1000*opti.variable(1,1);
    B_2 = 100*opti.variable(1,1);
        
    opti.set_initial(K_1, IG.K_1_sol);
    opti.set_initial(B_1,  IG.B_1_sol);
    opti.set_initial(K_2, IG.K_2_sol);
    opti.set_initial(B_2,  IG.B_2_sol);
    opti.set_initial(X, IG.X_sol);
    opti.set_initial(dX, IG.dX_sol);
    opti.set_initial(F_1, IG.F_1_sol);
    opti.set_initial(F_2, IG.F_1_sol);
    
%     opti.set_initial(K_1,-1000);
%     opti.set_initial(B_1, -100);
%     opti.set_initial(K_2, 1000);
%     opti.set_initial(B_2, 100);
%     opti.set_initial(X, pi);
%     opti.set_initial(dX, 0);
    
    opti.set_initial(F_1, 1);
    opti.set_initial(F_2, 1);
    
    opti.set_initial(L,   [L_scaling;0.1;L_scaling]);
    
    opti.subject_to(-1500 < K_1 < 0);
    opti.subject_to(-600 < B_1 < 0);
    
    opti.subject_to(1600 > K_2 > 0);
    opti.subject_to(600 > B_2 > 0);
%     
%     opti.subject_to(L(1,:)/L_scaling > 0);
%     opti.subject_to(L(3,:)/L_scaling > 0);
    
    
    
    Lk0 = [L(1,1) 0; L(2,1) L(3,1)];
    Pk0 = Lk0*Lk0';
    
    F_control_1 = F_1(:,1) + K_1*(X(:,1)-pi) + B_1*dX(:,1);
    F_control_1 = smoothClipping_tanh(F_control_1,sharpnessFactor,lowerClippingValue,upperClippingValue);
    
    F_control_2 = F_2(:,1) + K_2*(X(:,1)-pi) + B_2*dX(:,1);
    F_control_2 = -smoothClipping_tanh(F_control_2,sharpnessFactor,lowerClippingValue,upperClippingValue);
    
    dfdx_1 = fcn_derivative_smoothClipping_tanh(F_1,X,dX,K_1,B_1);
    dfdx_2 = fcn_derivative_smoothClipping_tanh(F_2,X,dX,K_2,B_2);
    
    
    obj = (F_control_1)^2 + (F_control_2)^2 + dfdx_1*Pk0*dfdx_1' + dfdx_2*Pk0*dfdx_2';
    
    k=1;
    
    %%% Local mesh variables
    F_1k =F_1(:,k);
    F_2k =F_2(:,k);
    Xdotk = fcn_OL(dX(:,k),X(:,k),F_1k,F_2k,0,K_1,B_1,K_2,B_2);
    opti.subject_to(Xdotk == 0);
    
    Ak = A_fcn(dX(:,k),X(:,k),F_1k,F_2k,0,K_1,B_1,K_2,B_2);
    Ck = C_fcn(dX(:,k),X(:,k),F_1k,F_2k,0,K_1,B_1,K_2,B_2);
    Pdotk = Ak*Pk0 + Pk0*Ak' + Ck*noiseVariance*Ck';
    opti.subject_to( [Pdotk(1,1);Pdotk(1,2);Pdotk(2,2)] == 0 );
    
    % Initial State
    opti.subject_to(X(:,1) == pi); % Position and velocity zero in initial state
    opti.subject_to(dX(:,end) ==  0); % Position and velocity zero in initial state
    
    %     opti.subject_to(((30*pi/180)^2 - Pk0(1,1))/L_scaling^2 > 0);
    %     opti.subject_to(((60*pi/180)^2 - Pk0(2,2))/L_scaling^2 > 0);
    
    opti.minimize(obj);
    
    
    % Solve
%         optionssol.ipopt.hessian_approximation = 'limited-memory';
    optionssol.ipopt.nlp_scaling_method = 'none';
    optionssol.ipopt.linear_solver = 'ma57';
    optionssol.ipopt.tol = 1e-7;
    optionssol.ipopt.constr_viol_tol = 1e-8;
    % optionssol.ipopt.nlp_scaling_max_gradient = 100;
    optionssol.ipopt.max_iter = 10000;
    result = solve_NLPSOL(opti,optionssol);
    % opti.callback(@(i) opti.debug.show_infeasibilities(1e-1));
    
    opti.solver('ipopt',optionssol)

    X_sol = result(1);
    dX_sol = result(2);
    
    F_1_sol = result(3);
    F_2_sol = result(4);
    
    L_sol = [L_scaling;L_scaling;L_scaling].*result(5:7);   
    
    K_1_sol = 1000*result(8);
    B_1_sol = 100*result(9);

    K_2_sol = 1000*result(10);
    B_2_sol = 100*result(11);
    
    
    F_control_1 = F_1_sol(:,1) + K_1_sol*(X_sol(:,1)-pi) + B_1_sol*dX_sol(:,1);
    F_control_1 = smoothClipping_tanh(F_control_1,sharpnessFactor,lowerClippingValue,upperClippingValue);
    
    F_control_2 = F_2_sol(:,1) + K_2_sol*(X_sol(:,1)-pi) + B_2_sol*dX_sol(:,1);
    F_control_2 = -smoothClipping_tanh(F_control_2,sharpnessFactor,lowerClippingValue,upperClippingValue);
    

    
    u_ff_sol = [F_control_1;F_control_2];
    
    pendulumResult_UKF.X_sol = X_sol;
    pendulumResult_UKF.dX_sol = dX_sol;
    pendulumResult_UKF.F_1_sol = F_1_sol;
    pendulumResult_UKF.F_2_sol = F_2_sol;
    
    pendulumResult_UKF.L_sol = L_sol;
    
    pendulumResult_UKF.K_1_sol = K_1_sol;
    pendulumResult_UKF.B_1_sol = B_1_sol;
    pendulumResult_UKF.K_2_sol = K_2_sol;
    pendulumResult_UKF.B_2_sol = B_2_sol;
    
    pendulumResult_UKF.u_ff_sol = u_ff_sol;
    
    obj_feedforward = sumsqr(u_ff_sol);
    
    
    dfdx_1 = fcn_derivative_smoothClipping_tanh(F_1_sol,X_sol(1),dX_sol(1),K_1_sol,B_1_sol);
    dfdx_2 = fcn_derivative_smoothClipping_tanh(F_2_sol,X_sol(1),dX_sol(1),K_2_sol,B_2_sol);
    
    
    obj = (F_control_1)^2 + (F_control_2)^2 + dfdx_1*Pk0*dfdx_1' + dfdx_2*Pk0*dfdx_2';
    
    P = [L_sol(1,k) 0; L_sol(2,k) L_sol(3,k)]*[L_sol(1,k) 0; L_sol(2,k) L_sol(3,k)]';
    [~,p] = chol(P);
    dfdx_1 = fcn_derivative_smoothClipping_tanh(F_1_sol,X_sol(1),dX_sol(1),K_1_sol,B_1_sol);
    dfdx_2 = fcn_derivative_smoothClipping_tanh(F_2_sol,X_sol(1),dX_sol(1),K_2_sol,B_2_sol);
    
    
    obj_corrections = full(dfdx_1*P*dfdx_1' + dfdx_2*P*dfdx_2');
    if p>0
        print('error')
    end
    
    pendulumResult_UKF.F_control_1_SD = sqrt([K_1_sol B_1_sol]*P*[K_1_sol; B_1_sol]);
    pendulumResult_UKF.F_control_2_SD = sqrt([K_2_sol B_2_sol]*P*[K_2_sol; B_2_sol]);
    
    pendulumResult_UKF.obj_corrections = obj_corrections;
    pendulumResult_UKF.obj_feedforward = obj_feedforward;
    pendulumResult_UKF.P_sol = P;
    pendulumResult_UKF.noiseVariance = noiseVariance;
    
    save(['EKFClipped_Variance' num2str(noiseVariance) '.mat'],'pendulumResult_UKF');
    
end
