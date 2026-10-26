function Centr = my_canopy( data )

%%%%%%%%%%%%%%% 加载数据 %%%%%%%%%%%%%%%%%%

Y = data(:,end);
[partion,~,~] = unique(Y);
class_num = length(partion);
[~,X_dim]=size(data);
data=data(:,1:X_dim-1);
[num,dim] = size(data);

k=zeros(class_num,1);
for t=1:class_num
    %%%%%%%%%%%%%%% 抽样 %%%%%%%%%%%%%%%%%%
    sample=round(num/10);
    rand_array=randperm(num);
    X_part=data(rand_array(1:sample),:);
    D=pdist(X_part);
    miu=mean(D);
    sigma=std(D);
    T2=miu+5*sigma;
    %%%%%%%%%canopy 自动划分聚类中心和个数%%%%%%%%%
    k(t) = 0;
    YB=[data zeros(num,1)];
    Centr=zeros(class_num,dim);
    while size(YB,1)&& (k(t)<class_num)
        k(t)=k(t)+1;
        Centr(k(t),:)=YB(1,1:dim);
        YB(1,:)=[];            %在选取第一个点为聚类点并删除
        L=size(YB,1);
        if L
            dist1=(YB(:,1:dim)-ones(L,1)*YB(1,1:dim)).^2;   %计算欧式距离
            dist2=sum(dist1,2);
        end
        for i=1:L-1
            if(dist2(i)<T2)  %<T2说明是该类，在矩阵中删除
                YB(i,4)=1;
            end
        end
        YB(YB(:,4)==1,:)=[];  %删除已归类的元素
    end
end


end

