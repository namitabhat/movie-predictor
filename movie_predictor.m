trg_path = 'path to training data file';
test_path = 'path to testing data file';

trg_data = load_data(trg_path);
test_data = load_data(test_path);
u_max = 2000;
m_max = 1500;


% calculate user and movie bias from training data 
% x = 1x(u_max+m_max) vector
A = construct_adjacency(trg_data,u_max,m_max);
nr_ratings = length(trg_data);
r_avg =  mean(trg_data(:,3));
y = trg_data(:,3)-r_avg;
x= A\y;

% calculate rmse of training data 
E = y - A*x;
trg_rmse  = calculate_rmse(E);
fprintf('RMSE (training) = %.3f\n',trg_rmse);

% get error matrix R for trg data
% error = actual rating - predicted rating
% R = u_max by m_max matrix of error values
r_predicted = zeros(nr_ratings,1);
for i=1:nr_ratings
    r_predicted(i) = r_avg + x(trg_data(i,1)) + x(trg_data(i,2)+u_max);
end

r_predicted(r_predicted>5)=5;
r_predicted(r_predicted<1)=1;
error = trg_data(:,3)-r_predicted;
R = spalloc(u_max,m_max,nr_ratings);

for i=1:nr_ratings
    R(trg_data(i,1),trg_data(i,2)) = error(i);
end



% calculate rmse of test data using baseline predictor
A = construct_adjacency(test_data,u_max,m_max);
nr_ratings = length(test_data);
r_avg =  mean(test_data(:,3));
y = test_data(:,3)-r_avg;
E = y - A*x;
test_rmse  = calculate_rmse(E);
fprintf('RMSE (test) = %.3f\n',test_rmse);



% calculate rmse of test data using neighbourhood model predictor 

% get movie-movie similarity and rank top L similar movies for each movie 
D = get_similarity(R,m_max);
[~,I] = sort(abs(D),2,'descend');
L = 750;										
I = I(:,1:L);

% get absolute value of similarity for top L movies 
denom = zeros(m_max,L);
for k = 1:m_max
	denom(k,:) = abs(D(k,I(k,:))); 
end

% apply neighbourhood model to get improved prediction (ri_predicted)
ri_predicted = zeros(nr_ratings,1);
udata = test_data(:,1);
mdata = test_data(:,2);
for l = 1:nr_ratings
    
    % get D(i,j)*R(u,j)
	tmp = R(udata(l),:)' .* D(:, mdata(l));			
	tmp = tmp(I(mdata(l),:), :);
	num = sum(tmp);
    
    % sum of absolute D(i,j) values for all j movies 
	tmp(~tmp==0)=1;					
	denominator = sum(denom(mdata(l),:).*tmp');		
    
    % add improved prediction to base prediction
    baseline = r_avg + x(udata(l)) + x(u_max+mdata(l));
    if (denominator ~= 0)
		ri_predicted(l) = baseline + num/denominator;	
    else
        ri_predicted(l) = baseline;
    end
end
    
ri_predicted(ri_predicted>5)=5;
ri_predicted(ri_predicted<1)=1;
rmse_improved = calculate_rmse(test_data(:,3)-ri_predicted);
fprintf('RMSE (improved) = %.3f\n',rmse_improved);




function data=load_data(path)
    fid = fopen(path,'r');
    eob = textscan(fid,'%f,%f,%f','delimiter','\n');
    data = cell2mat(eob);
    fclose(fid);
end

function A=construct_adjacency(data,u_max,m_max)
    nr_ratings = length(data);
    udata = data(:,1);
    mdata = data(:,2);
    A = sparse([1:nr_ratings 1:nr_ratings]', [udata; mdata+u_max], ones(2*nr_ratings,1),nr_ratings,u_max+m_max);
end

function rmse=calculate_rmse(diff)
    rmse = sqrt(mean(diff.^2));
end


function D=get_similarity(R,max)
    v = ones(1,max);
    D = diag(v);
    for i=1:max
       for j=1:max
          if (i==j)
              continue;
          end
          users = find(R(:,i) & R(:,j));
          if length(users)>10
              num=0;
              denom_i=0;
              denom_j=0;
              for u=1:length(users)
                num = num + R(users(u),i)*R(users(u),j);
                denom_i = denom_i + R(users(u),i)^2;
                denom_j = denom_j + R(users(u),j)^2;
              end
              D(i,j) = num/sqrt(denom_i*denom_j);
          end
       end
    end
end






