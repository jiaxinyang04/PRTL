function [Out,obj] = PRTL(X, cls_num, gt, opts)
    %% Parameter settings
    N = size(X{1}, 2); % the number of samples
    K = length(X);     % the number of views
    epsilon = 1e-7;
    tau = 1e-5;        % penalty factor
    miu2 = 1e-5;     
    max_tau = 1e10;
    max_miu2 = 1e10;
    if  isfield(opts, 'maxIter');       maxIter = opts.maxIter;         end
    if  isfield(opts, 'yita');          yita = opts.yita;               end
    if  isfield(opts, 'mul_rate');      mul_rate = opts.mul_rate;       end
    if  isfield(opts, 'nb_num');        nb_num = opts.nb_num;           end
    %% Initialize...
    for k=1:K
        Z{k} = zeros(N,N); 
        M{k} = zeros(N,N);
        G{k} = zeros(N,N);
        Q{k} = zeros(N,N);
        B{k} = zeros(N,N);
        L{k} = zeros(N,N);
        P{k} = zeros(size(X{k},1),size(X{k},1));
        XXT{k}=X{k}*X{k}';
        XTX{k}=X{k}'*X{k};
        XXTI{k}=pinv(X{k}*X{k}'+eye(size(X{k},1),size(X{k},1)));
    end
    iter = 1;
    Isconverg = 0;
    while(Isconverg == 0)
        %% ------------------- Update L^k -------------------------------
        for k=1:K
            Weight{k} = my_constructW_PKN((abs(Z{k})+abs(Z{k}'))./2, nb_num);
            Diag_tmp = diag(sum(Weight{k}));
            L{k} = Diag_tmp - Weight{k};
        end   
        %% ------------------- Update Z^k -------------------------------    
        for k=1:K
            tmp = (tau*G{k}-M{k})+2*XTX{k}+(miu2*Q{k}-B{k})-2*X{k}'*P{k}*X{k};
            linshi_Z = (2*XTX{k}+(tau+miu2)*eye(N,N))\tmp;
            for ic = 1:size(Z{k},1)
                ind = 1:size(Z{k},2);
                Z{k}(ic,ind) = EProjSimplex_new(linshi_Z(ic,ind));
                Z{k}(isnan(Z{k})) = 0; Z{k}(isinf(Z{k})) = 0;
            end
        end
       %% ------------------- Update P^k -------------------------------
        for k=1:K
            linshi_P = (XXT{k}-X{k}*Z{k}*X{k}')*XXTI{k};
            for ic = 1:size(P{k},1)
                ind = 1:size(P{k},2);
                P{k}(ic,ind) = EProjSimplex_new(linshi_P(ic,ind));
                P{k}(isnan(P{k})) = 0; P{k}(isinf(P{k})) = 0;
            end
        end
        %% ------------------- Update Q^k -------------------------------
        for k=1:K
            Q{k} = (miu2*Z{k} + B{k})/((L{k}+L{k}') + miu2*eye(N,N));
            Q{k}(isnan(Q{k})) = 0; Q{k}(isinf(Q{k})) = 0;
        end
        %% ------------------- Update G ---------------------------------
        Z_tensor = cat(3, Z{:,:});
        M_tensor = cat(3, M{:,:});
        zv = Z_tensor(:);
        mv = M_tensor(:);
        clear M
        [gv, ~] = wshrinkObj2(zv+1/tau*mv, 1/tau, [N, N, K], 0, 3,yita);
        G_tensor = reshape(gv, [N, N, K]);
        for k=1:K
            G{k} = G_tensor(:,:,k);
        end
        %% ------------------- Update auxiliary variables ---------------
        M_tensor = M_tensor  + tau*(Z_tensor - G_tensor);
        for k=1:K
            M{k} = M_tensor(:,:,k);
            B{k} = B{k} + miu2*(Z{k}-Q{k});  
        end   
        %% ------------------- Update penalty params --------------------
        tau = min(tau*mul_rate, max_tau);
        miu2 = min(miu2*mul_rate, max_miu2);
        %% ------------------- Converge check ---------------------------
        Isconverg = 1;%1停0不停
        history.norm_Z_G(iter) = max(cellfun(@(z,g) norm(z-g, inf), Z, G));
        if (history.norm_Z_G(iter)>epsilon)      
            Isconverg = 0;
        end    
        history.norm_Z_Q(iter) = max(cellfun(@(z,q) norm(z-q, inf), Z, Q));
        if (history.norm_Z_Q(iter)>epsilon)   
            Isconverg = 0;
        end 
        if (iter>maxIter)
            Isconverg  = 1;
        end
        %% each term obj
        term1(iter)=0;term2(iter)=0;term3(iter)=0;term4(iter)=0;term5(iter)=0;
        for k = 1:K
            term1(iter) = term1(iter)+norm(X{k}-X{k}*Z{k}-P{k}*X{k},'fro')^2;
            term2(iter) = term2(iter)+trace(Z{k}*L{k}*Z{k}');
            term3(iter) = term3(iter)+norm(P{k},'fro')^2;
            term4(iter) = term4(iter)+norm(Z{k}-G{k},'fro')^2;
            term5(iter) = term5(iter)+norm(Z{k}-Q{k},'fro')^2;
        end
        obj(iter) = max([term1(iter),term2(iter),term3(iter),term4(iter),term5(iter)]); 
        if mod(iter, 10) == 0
            fprintf('iter: %d  ', iter);
        end
        iter = iter + 1;
    end
    %% ---------------- Clustering --------------------------------------
    S = 0;
    for k=1:K
        S = S + abs(Z{k})+abs(Z{k}');
    end
    C = SpectralClustering(S,cls_num);
    ACC = Accuracy(C,double(gt));
    [~, nmi, ~] = compute_nmi(gt,C);
    [~,~, PUR] = purity(gt,C);
    [f,p,r] = compute_f(gt,C);
    [AR,~,~,~]=RandIndex(gt,C);
    %% ---------------- Record ------------------------------------------
    Out.ACC = ACC;
    Out.NMI = nmi;
    Out.PUR = PUR;
    Out.AR = AR;
    Out.recall = r;
    Out.precision = p;
    Out.fscore = f;
    Out.history = history;
end


