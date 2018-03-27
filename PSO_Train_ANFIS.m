% Author:   KASHIF HUSSAIN, PhD. in Machine Learning and Metaheuristics,
%           Universiti Tun Hussein Onn Malaysia, Johor, Malaysia.
% Email:    usitsoft@hotmail.com
% ANFIS trained by PSO for Iris Classification Problem
% No. of inputs = 4, No. of output = 1, No. of MFs in each input = 2 
% Total Rules = 2^4=16 (Grid Partitioning)
% Total PSO particles = 15
% No. of parameters in each particle = 96
% Maximum Iterations = 20
% Error Tolerance = 0.05
% Training Set = 100
% Testing Set = 50

function Main()
    close all
    clear all 
    clc
    
    fid = fopen('IrisTraining.dat','r');
    Training_data = textscan(fid, '%f%f%f%f%f');
    SepalLength   = Training_data{1}; % SepalLength
    SepalWidth    = Training_data{2}; % SepalWidth
    PetalLength   = Training_data{3}; % PetalLength
    PetalWidth    = Training_data{4}; % PetalWidth
    Target = Training_data{5}/3; % Target = Class of Iris
    
    % Training ANFIS
    [TrainingMSE, bestParams, itr] = PSO_Train(SepalLength, SepalWidth, PetalLength, PetalWidth, Target);
    TrainingAcc = ((1 - TrainingMSE / 2) * 100);

    % Test ANFIS
    fid = fopen('IrisTesting.dat','r');
    Training_data = textscan(fid, '%f%f%f%f%f');
    SepalLength     = Training_data{1}; % SepalLength
    SepalWidth     = Training_data{2}; % SepalWidth
    PetalLength     = Training_data{3}; % PetalLength
    PetalWidth     = Training_data{4}; % PetalWidth
    Target = Training_data{5}/3; % Target = Class of Iris
    sse = 0;
    nans = 0;
    for t = 1: size(Target, 1)
         outputANFIS = ANFIS_GetOutput(bestParams, SepalLength(t), SepalWidth(t), PetalLength(t), PetalWidth(t));
         se = (Target(t) - outputANFIS)^2; 
        if isnan(se), 
            nans = nans + 1;
            continue
        end
        sse = sse + se; % sum of squared error
    end
    TestingMSE = sse/(size(Target, 1)-nans);
    TestingAcc = ((1 - TestingMSE / 2) * 100);
    msgbox({'Iris', ...
        ['Training MSE: ',num2str(TrainingMSE),' Training Accuracy: ',num2str(TrainingAcc),'%'], ...
        ['Testing MSE: ',num2str(TestingMSE),' Testing Accuracy: ',num2str(TestingAcc),'%'], ...
        ['Iterations to converge: ',num2str(itr)] ...
        });
end    

function [TrainingMSE, bestParams, Iterations] = PSO_Train(SepalLength, SepalWidth, PetalLength, PetalWidth, Target)
    totalParam = 96; % 16*5 + 8*2 = 96
    maxIterations = 20;
    itr = 1;
    errTolerance = 0.05;
    
    % PSO initialization
    %--------------------------------------------------------
    fbest = 1.0e+100;   % Global best
    totalParticles = 15; % No. of particles
    Ub = 1 * ones(totalParticles,totalParam);
    Lb = -1 * ones(totalParticles,totalParam);

    minInertiaWeight = 0.4;
    maxInertiaWeight = 0.9;
    socialConst = 2;
    cognitiveConst = 2;
    velClampingFactor = 2;
    pbestval = 1.0e+100*ones(1,totalParticles);    % Personal best values
    particles = init_Swarm(totalParticles, Lb(1,:), Ub(1,:), totalParam);
    velocity = init_Velocity(totalParticles, totalParam, velClampingFactor, Ub(1,:));
    %--------------------------------------------------------

    % Execute PSO
    %--------------------------------------------------------
    while ((fbest > errTolerance) && (itr <= maxIterations))
        for i=1:totalParticles
            sse = 0;
            nans = 0;
            for t = 1: size(Target, 1)
                outputANFIS = ANFIS_GetOutput(particles(i,:), SepalLength(t), SepalWidth(t), PetalLength(t), PetalWidth(t));
                se = (Target(t) - outputANFIS)^2; % SE = Squared Error
                if isnan(se), 
                    nans = nans + 1;
                    continue
                end                
                sse = sse + se; % Sum of Squared Errors
            end
            fval = sse/(size(Target, 1)-nans); % MSE = Mean Squared Error
            
            % Personal Best
            if isnan(fval) || fval<=pbestval(i),
                pbest(i,:) = particles(i,:);
                pbestval(i) = fval;
            end
            
            % Global best
            if fval<=fbest,
                gbest = particles(i,:);
                fbest=fval;
            end             
        end
        
        % update velocity
        w=((maxIterations - itr)*(maxInertiaWeight - minInertiaWeight))/(maxIterations-1) + minInertiaWeight;
        velocity = pso_velocity(totalParticles, totalParam, velocity, gbest, pbest, particles, w, socialConst, cognitiveConst, velClampingFactor, Ub);
        % update position
        particles = pso_move(particles,velocity,Lb,Ub);
        itr = itr + 1;
    end
    %--------------------------------------------------------
     
    TrainingMSE = fbest;
    bestParams = gbest;
    Iterations = itr-1;
end

function outputANFIS = ANFIS_GetOutput(params, SepalLength, SepalWidth, PetalLength, PetalWidth)
    numOfInputs = 4;
    numOfMFTerms = 2;
    numOfRules = numOfMFTerms^numOfInputs;
    
    %=====================================================
    % Layer 0: Input Layer, Input variable names
    %=====================================================
    % 1- SepalLength
    fis.input(1).name = ['input' 'SepalLength'];
    fis.input(1).range=[4.30 7.90];
    fis.input(1).value= SepalLength;
    % 2- SepalWidth
    fis.input(2).name = ['input' 'SepalWidth'];
    fis.input(2).range=[2.00 4.40];
    fis.input(2).value= SepalWidth;
    % 3- PetalLength
    fis.input(3).name = ['input' 'PetalLength'];
    fis.input(3).range=[1.00 6.90];
    fis.input(3).value= PetalLength;
    % 4- PetalWidth
    fis.input(4).name = ['input' 'PetalWidth'];
    fis.input(4).range=[0.10 2.50];
    fis.input(4).value= PetalWidth;

    %=====================================================
    % Layer 1: Membership functions layer
    %=====================================================
    weightIndex = 1;
    for i=1:numOfInputs
        for j=1:numOfMFTerms
            fis.input(i).mf(j).params = [params(weightIndex) params(weightIndex+1)];
            fis.input(i).mf(j).MD =  gmf(double(fis.input(i).value), double(fis.input(i).mf(j).params(2)), double(fis.input(i).mf(j).params(1)));
            weightIndex = weightIndex + 2;
        end
    end

    %=====================================================
    % Layer 2a: Product Layer, Initialize rules list. Prod(membership degrees of all inputs)
    %=====================================================
    fis.rule=[];
    for i=1:numOfRules
        fis.rule(i).antecedent=zeros(1:2); % membership degrees of SepalLength, SepalWidth, PetalLength,...,xn
        fis.rule(i).prod=1; % product of all antecedents of each rule
        fis.rule(i).norm=0; % w' = weight of this rule
        fis.rule(i).consequent=0; % w'.f
    end

    % ==========================================================
    % Layer 2b: Grid Partitioning: Create all possible combinations of inputs and input terms
    % ==========================================================
    for i=1:numOfInputs 
        if i<numOfInputs
            j=1;
            for m=0: numOfMFTerms^(numOfInputs-i):numOfMFTerms^numOfInputs - numOfMFTerms^(numOfInputs-i)
                if (j<=numOfMFTerms)
                    for l=1:numOfMFTerms^(numOfInputs-i)
                         fis.rule(m+l).antecedent(i) = fis.input(i).mf(j).MD;   % m_out(m+l,i) = m_out(m+l,i) = m_in(i,j);
                    end
                else 
                   j=1;
                    for l=1:numOfMFTerms^(numOfInputs-i)
                         fis.rule(m+l).antecedent(i) = fis.input(i).mf(j).MD;
                    end
                end     
                j = j + 1;
            end
        elseif i == numOfInputs
            for m=0: numOfMFTerms: (numOfMFTerms^numOfInputs)-numOfMFTerms 
                j=1;
                for l=1: numOfMFTerms 
                    fis.rule(m+l).antecedent(i) = fis.input(i).mf(j).MD;
                    j = j + 1;
                end
            end
       end 
    end
    
    %=====================================================
    % Layer 2c: Product all antecedents of each rule
    %=====================================================
    SumOfAllRules = 0;
    for i=1:numOfRules
        for j=1:length(fis.rule(i).antecedent)
            fis.rule(i).prod = fis.rule(i).prod * fis.rule(i).antecedent(j);
        end
        SumOfAllRules = SumOfAllRules + fis.rule(i).prod;
    end

    %=====================================================
    % Layer 3: Normalization Layer: Normalize each rule
    %=====================================================
    for i=1:numOfRules
        fis.rule(i).norm = fis.rule(i).prod / SumOfAllRules;
    end    
    
    %=====================================================
    % Layer 4: Output Membership Functions, Constant/Linear
    %=====================================================
    outputRules = 0;
    for i = 1:numOfRules
        fis.rule(i).consequent = fis.rule(i).norm * ((SepalLength*params(weightIndex))+(SepalWidth*params(weightIndex+1))+(PetalLength*params(weightIndex+2))+(PetalWidth*params(weightIndex+3))+params(weightIndex+4));
        outputRules = outputRules + fis.rule(i).consequent;
        weightIndex = weightIndex + 5;    
    end
    
    %=====================================================
    % Layer 5: Output Membership Functions, Constant/Linear
    %=====================================================    
    outputANFIS = outputRules;
end   

function [guess]=init_Swarm(n,Lb,Ub,ndim)
    for i=1:n,
        guess(i,1:ndim)=Lb+rand(1,ndim).*(Ub-Lb);
    end
end

function [vel] = init_Velocity(totalParticles, ndim, velClampingFactor, Ub)
    vMax = Ub*velClampingFactor;
    vMin = -vMax;
    for i=1:totalParticles
        vel(i,:)=vMin+(vMax-vMin).*rand(1,ndim);
    end
end

function velocity = pso_velocity(totalParticles, ndim, vel,gbest,pbest,particle,w, socialConst, cognitiveConst, velClampingFactor, Ub)
    vMax = Ub*velClampingFactor;
    vMin = -vMax;
    for i=1:totalParticles,
        velocity(i,:) = w*vel(i,:) + socialConst*rand(1,ndim).*(gbest-particle(i,:)) + cognitiveConst*rand(1,ndim).*(pbest(i,:)-particle(i,:));                        
    end
    velocity=findrange(velocity,vMin,vMax);
end

function particle = pso_move(best,vel,Lb,Ub)
    totalParticles=size(best,1);
    for i=1:totalParticles,
        particle(i,:) = best(i,:)+vel(i,:);
    end
    particle=findrange(particle,Lb,Ub);
end

function particles=findrange(part, Lb, Ub)
    totalParticles=size(part,1);
    for i=1:totalParticles,
        part(i,:)=min(part(i,:),Ub(i,:));
        part(i,:)=max(part(i,:),Lb(i,:));
    end
    particles = part;
end

function md = gmf(x,c,sigma)
    %md = x+c+sigma;
    md = exp(-(((x - c)/sigma)^2)/2);
end

