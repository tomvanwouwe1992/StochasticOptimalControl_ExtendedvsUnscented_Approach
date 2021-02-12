import casadi.*
import org.opensim.modeling.*
clear all; close all; clc;
figure(1)
cs = linspecer(15);
dt = 0.01;

noiseVariances = [1 2 5 10 20 50 100 200 500 1000 2000 5000 ];

for i =1:1
    
    noiseVariance = noiseVariances(i); %N²s
    
    noiseVarianceDiscrete = noiseVariance/dt;
    noiseStdDiscrete = sqrt(noiseVarianceDiscrete);
    
        load(['UKFClipped_Variance' num2str(noiseVariances(i)) '.mat']);
        IG = pendulumResult_UKF;
    L_scaling = 1e-2;
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
    
    
    
    %% Setting up OCP
    %- Time parameters
    options.tf = dt;
    options.t0 = 0;
    h = dt;
    time_vector = options.t0:dt:options.tf;
    time = time_vector;
    
    %- Create collocation integration scheme with options
    options.grid = options.t0:dt:options.tf;
    options.number_of_finite_elements = round(options.tf/dt);
    N = options.number_of_finite_elements;
    time_vector = options.t0:dt:options.tf;
    
    
    %- Stochastic Dynamics of the controlled system
    % Variables initialisaiton
    X = MX.sym('X',1,1);
    dX = MX.sym('dX',1,1);
    F_dist =  MX.sym('F_dist',1,1);
    F_1 =  MX.sym('F_1',1,1);
    F_2 =  MX.sym('F_2',1,1);
    
    F_min =  MX.sym('F_min',1,1);
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
    F_control_2_clipped = -smoothClipping_tanh(F_control_2,sharpnessFactor,lowerClippingValue,upperClippingValue);
    
    Xdot = [ dX; (-m*g*l*sin(X) + F_control_1_clipped + F_control_2_clipped + F_dist)/(m*l^2)];
    fcn_OL = Function('fcn_OL',{dX,X,F_1,F_2,F_dist,K_1,B_1,K_2,B_2},{Xdot});
    
    
    fcn_smoothClipping_tanh = Function('fcn_smoothClipping_tanh',{F_1,X,dX,K_1,B_1},{F_control_1_clipped});
    
    derivative_smoothClipping_tanh = jacobian(F_control_1_clipped,[X dX]);
    fcn_derivative_smoothClipping_tanh = Function('fcn_derivative_smoothClipping_tanh',{F_1,X,dX,K_1,B_1},{derivative_smoothClipping_tanh});
    
    
    
    clear X dX F K B
    
    %- Variables and controls
    opti = casadi.Opti();
    
    
    %% States/Controls of the nominal system
    X = opti.variable(1,(N+1));
    dX = opti.variable(1,(N+1));
    F_1 = opti.variable(1,1);
    F_2 = opti.variable(1,1);
    
    opti.subject_to(F_1 > 1e-3);
    opti.subject_to(F_2 > 1e-3);
    
    L = [L_scaling;L_scaling;L_scaling].*opti.variable(3,(N+1));
    K_1 = 1000*opti.variable(1,1);
    B_1 = 100*opti.variable(1,1);
    K_2 = 1000*opti.variable(1,1);
    B_2 = 100*opti.variable(1,1);
    Xdot = opti.variable(2,(2*nTOT+1)*(N));
    
%     opti.set_initial(K_1, -1000);
%     opti.set_initial(B_1,  -100);
%     opti.set_initial(K_2, 1000);
%     opti.set_initial(B_2,  100);
%     opti.set_initial(X, [pi pi]);
%     opti.set_initial(dX, [0 0]);
%     opti.set_initial(Xdot, zeros(2,7));
%     opti.set_initial(F_1, 8);
%     opti.set_initial(F_2, 8);

    opti.set_initial(K_1, IG.K_1_sol);
    opti.set_initial(B_1,  IG.B_1_sol);
    opti.set_initial(K_2, IG.K_2_sol);
    opti.set_initial(B_2,  IG.B_2_sol);
    opti.set_initial(X, IG.X_sol);
    opti.set_initial(dX, IG.dX_sol);
    opti.set_initial(Xdot, IG.Xdot_sol);
    opti.set_initial(F_1, IG.F_1_sol);
    opti.set_initial(F_2, IG.F_1_sol);
% IG
    
    opti.set_initial(L,   [L_scaling;L_scaling;L_scaling].*ones(3,N+1));
    
    opti.subject_to(-1500 < K_1 < 0);
    opti.subject_to(-500 < B_1 < 0);
    
    opti.subject_to(1500 > K_2 > 0);
    opti.subject_to(500 > B_2 > 0);
    
%     opti.subject_to(L(1,:)/L_scaling > 0);
%     opti.subject_to(L(3,:)/L_scaling > 0);
    
    
    
    Lk0 = [L(1,1) 0; L(2,1) L(3,1)];
    Pk0 = Lk0*Lk0';
    
    F_control_1 = F_1(:,1) + K_1*(X(:,1)-pi) + B_1*dX(:,1);
    F_control_1 = smoothClipping_tanh(F_control_1,sharpnessFactor,lowerClippingValue,upperClippingValue);
    
    F_control_2 = F_2(:,1) + K_2*(X(:,1)-pi) + B_2*dX(:,1);
    F_control_2 = -smoothClipping_tanh(F_control_2,sharpnessFactor,lowerClippingValue,upperClippingValue);
    
    dfdx_1 = fcn_derivative_smoothClipping_tanh(F_1,X(1),dX(1),K_1,B_1);
    dfdx_2 = fcn_derivative_smoothClipping_tanh(F_2,X(1),dX(1),K_2,B_2);
    
    
    obj = (F_control_1)^2 + (F_control_2)^2 + dfdx_1*Pk0*dfdx_1' + dfdx_2*Pk0*dfdx_2';
    
    
    for k = 1:N
        %%% Local mesh variables
        F_1k =F_1(:,k);
        F_2k =F_2(:,k);
        
        Lk = [L(1,k) 0 0; L(2,k) L(3,k) 0; 0 0 noiseStdDiscrete];
        Ak = c*Lk;
        Yk = [[X(:,k); dX(:,k)]*ones(1,nStates+nNoiseSources);  zeros(1,1)*ones(1,nStates+nNoiseSources)];
        sigmask = [Yk(:,1) Yk + Ak  Yk - Ak];
        
        Xdotk = fcn_OL(sigmask(2,:),sigmask(1,:),F_1k.*ones(1,7),F_2k.*ones(1,7),sigmask(3,:),K_1,B_1,K_2,B_2);
        Xdotk_plus = Xdot(:,(k-1)*(2*nTOT+1)+1:(k)*(2*nTOT+1));
        sigmask_next = sigmask(1:2,:) + 0.5*h*Xdotk + 0.5*h*Xdotk_plus;
        opti.subject_to(Xdotk_plus - fcn_OL(sigmask_next(2,:),sigmask_next(1,:),F_1k.*ones(1,7),F_2k.*ones(1,7),sigmask(3,:),K_1,B_1,K_2,B_2) == 0);
        
        F_1_clipped = fcn_smoothClipping_tanh(F_1k.*ones(1,7),sigmask(1,:),sigmask(2,:),K_1,B_1);
        
        F_2_clipped = fcn_smoothClipping_tanh(F_2k.*ones(1,7),sigmask(1,:),sigmask(2,:),K_2,B_2);
        
        effort = F_1_clipped.^2 + F_2_clipped.^2;
        
        expected_effort = effort*WeightVec_M';
        
        mean_next = sigmask_next*WeightVec_M';
        opti.subject_to(mean_next - [X(:,k+1); dX(:,k+1)] == 0);
        
        Y1k = sigmask_next - mean_next;
        Pk_next = Y1k*diag(WeightVec_C)*Y1k';
        Lk_next = [L(1,k+1) 0; L(2,k+1) L(3,k+1)];
        Pk_next_estimated = Lk_next*Lk_next';
        opti.subject_to([Pk_next_estimated(1,1) - Pk_next(1,1); Pk_next_estimated(1,2) - Pk_next(1,2); Pk_next_estimated(2,2) - Pk_next(2,2)]./[L_scaling.^2;L_scaling.^2;L_scaling.^2] == 0);
        
    end
    
    
    F_control_1 = F_1(:,1) + K_1*(X(:,2)-pi) + B_1*dX(:,2);
    F_control_1 = smoothClipping_tanh(F_control_1,sharpnessFactor,lowerClippingValue,upperClippingValue);
    
    F_control_2 = F_2(:,1) + K_2*(X(:,2)-pi) + B_2*dX(:,2);
    F_control_2 = -smoothClipping_tanh(F_control_2,sharpnessFactor,lowerClippingValue,upperClippingValue);
    
    dfdx_1 = fcn_derivative_smoothClipping_tanh(F_1,X(2),dX(2),K_1,B_1);
    dfdx_2 = fcn_derivative_smoothClipping_tanh(F_2,X(2),dX(2),K_2,B_2);
    
    
    obj = expected_effort; %obj + (F_control_1)^2 + (F_control_2)^2 + dfdx_1*Pk_next_estimated*dfdx_1' + dfdx_2*Pk_next_estimated*dfdx_2';
    % Initial State
    opti.subject_to(X(:,1) == pi); % Position and velocity zero in initial state
    opti.subject_to(X(:,end) ==  pi); % Position and velocity zero in initial state
    opti.subject_to(dX(:,1) == 0); % Position and velocity zero in initial state
    opti.subject_to(dX(:,end) == 0); % Position and velocity zero in initial state
    
    opti.subject_to(((30*pi/180)^2 - Pk_next_estimated(1,1))/L_scaling^2 > 0);
    opti.subject_to(((200*pi/180)^2 - Pk_next_estimated(2,2))/L_scaling^2 > 0);
    
    
    opti.subject_to((L(:,1) - L(:,2))./[L_scaling;L_scaling;L_scaling] == 0);
    
    opti.minimize(obj);
    
    
    % Solve
    %     optionssol.ipopt.hessian_approximation = 'limited-memory';
    
    optionssol.ipopt.nlp_scaling_method = 'none';
    optionssol.ipopt.linear_solver = 'ma57';
    optionssol.ipopt.tol = 1e-7;
    optionssol.ipopt.constr_viol_tol = 1e-8;
    optionssol.ipopt.max_iter = 10000;
    result = solve_NLPSOL(opti,optionssol);
    % opti.callback(@(i) opti.debug.show_infeasibilities(1e-1));
    
    opti.solver('ipopt',optionssol)
%     sol = opti.solve();
%     X_sol = sol.value(X);
%     dX_sol = sol.value(dX);
    
%     F_1_sol = sol.value(F_1);
%     K_1_sol = sol.value(K_1);
%     B_1_sol = sol.value(B_1);
%     F_2_sol = sol.value(F_2);
%     K_2_sol = sol.value(K_2);
%     B_2_sol = sol.value(B_2);
%     
%     L_sol = sol.value(L);
%     Xdot_sol = sol.value(Xdot);
    
    
        X_sol = result(1:2);
    dX_sol = result(3:4);
    
    F_1_sol = result(5);
    F_2_sol = result(6);
    
    L_sol = [L_scaling;L_scaling;L_scaling].*[result(7:9) result(10:12)];   
    
    K_1_sol = 1000*result(13);
    B_1_sol = 100*result(14);

    K_2_sol = 1000*result(15);
    B_2_sol = 100*result(16);
    
    Xdot_sol = reshape(result(17:end),2,7);
    
    F_control_1 = F_1_sol(:,1) + K_1_sol*(X_sol(1,1)-pi) + B_1_sol*dX_sol(1,1);
    F_control_1 = smoothClipping_tanh(F_control_1,sharpnessFactor,lowerClippingValue,upperClippingValue);
    
    F_control_2 = F_2_sol(:,1) + K_2_sol*(X_sol(:,1)-pi) + B_2_sol*dX_sol(:,1);
    F_control_2 = -smoothClipping_tanh(F_control_2,sharpnessFactor,lowerClippingValue,upperClippingValue);
    
    u_ff_sol = [F_control_1;F_control_2];
    
    
    Lk = [L_sol(1,1) 0 0; L_sol(2,1) L_sol(3,1) 0; 0 0 noiseStdDiscrete];
    Ak = c*Lk;
    Yk = [[X_sol(1,1); dX_sol(1,1)]*ones(1,nStates+nNoiseSources);  zeros(1,1)*ones(1,nStates+nNoiseSources)];
    sigmask = [Yk(:,1) Yk + Ak  Yk - Ak];

        
    pendulumResult_UKF.X_sol = X_sol;
    pendulumResult_UKF.dX_sol = dX_sol;
    pendulumResult_UKF.F_1_sol = F_1_sol;
    pendulumResult_UKF.F_2_sol = F_2_sol;
    
    pendulumResult_UKF.Xdot_sol = Xdot_sol;
    
    pendulumResult_UKF.L_sol = L_sol;
    
    pendulumResult_UKF.K_1_sol = K_1_sol;
    pendulumResult_UKF.B_1_sol = B_1_sol;
    pendulumResult_UKF.K_2_sol = K_2_sol;
    pendulumResult_UKF.B_2_sol = B_2_sol;
    
    pendulumResult_UKF.u_ff_sol = u_ff_sol;
    
    obj_feedforward = sumsqr(u_ff_sol);
    
    
    dfdx_1 = fcn_derivative_smoothClipping_tanh(F_1_sol,X_sol(1),dX_sol(1),K_1_sol,B_1_sol);
    dfdx_2 = fcn_derivative_smoothClipping_tanh(F_2_sol,X_sol(1),dX_sol(1),K_2_sol,B_2_sol);
    
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
    
    save(['UKFClipped_Variance' num2str(noiseVariance) '.mat'],'pendulumResult_UKF');
    
end

