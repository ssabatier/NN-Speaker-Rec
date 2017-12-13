% This code finds the respective twin pairs for identical and fraternal
% from all years

% Get Twins 2014 Audio
listRID=dir(fullfile('/home/stallone/Desktop/Twins2014Data/Audio')); % list of RIDs
ridPath = '/home/stallone/Desktop/Twins2014Data/Audio';
listRIDvec={};
% for l=3:length(listRID)
%     listRIDvec{l-2}=listRID(l).name;
% end

% Get Twin 2014 Relationships
metamat=strcat('/home/stallone/Desktop/Twins2014Data/metadataTwins2014.xlsx');  
[~,~,meta] = xlsread(metamat);
[row,col]=size(meta);
for m=2:row
     meta{m,1}=num2str(meta{m,1});
end

% Start Matching for Twins 2014
twinRelId14={};
twinRelFr14={};
id=1;
fr=1;
for j=2:length(meta)
    RID=meta{j,1}; % get current RID  
%     if ismember(RID,listRIDvec)
%         for j=2:length(meta)
            if j==2
               twinRelId14{id,1} = RID;
               TTID=meta{j,6};
               for k=j+1:length(meta)
                  if meta{k,2}==TTID
                     twinRelId14{id,2}=meta{k,1};                      
                  end
               end 
               id=id+1;
            elseif ~ismember(RID,twinRelId14(:,1)) && ~ismember(RID,twinRelId14(:,2)) && j~=185 && j~=188 % ~strcmp(RID,meta{j-1,1}) &&%&& meta{j,2}~=meta{j-1,6}
               if strcmp(meta{j,9},'Identical') || strcmp(meta{j,9},'Identical Mirror')
                  twinRelId14{id,1} = RID;
                  TTID=meta{j,6};
                  for k=j+1:length(meta)
                      if meta{k,2}==TTID
                         twinRelId14{id,2}=meta{k,1};                      
                      end
                  end 
                  id=id+1;
               else 
                   msg=sprintf('RID %s is not an identical twin',RID);
                   disp(msg)
                   if j==30
                       twinRelFr14{fr,1} = RID;
                       TTID=meta{j,6};
                      for k=j+1:length(meta)
                          if meta{k,2}==TTID
                             twinRelFr14{fr,2}=meta{k,1};                      
                          end
                      end
                       fr=fr+1;
                   elseif strcmp(meta{j,9},'Fraternal') && ~ismember(RID,twinRelFr14(:,1)) && ~ismember(RID,twinRelFr14(:,2)) && j~=83
                      twinRelFr14{fr,1} = RID;
                      TTID=meta{j,6};
                      for k=j+1:length(meta)
                          if meta{k,2}==TTID
                             twinRelFr14{fr,2}=meta{k,1};                      
                          end
                      end
                       fr=fr+1;
                   else 
                      msg=sprintf('RID %s is not fraternal or identical or is already present in Frat',RID);
                      disp(msg)
                   end
               end
            end   
%         end
%     else
%         msg=sprintf('RID %s is not in metadata 2014',RID);
%         disp(msg) 
%     end
end

%% Twin Matcher for 2015 
clc 
clear all

% Get Twin 2015 Audio
listRID=dir(fullfile('/home/stallone/Desktop/Twins2015Data/Audio')); % list of RIDs  
ridPath = '/home/stallone/Desktop/Twins2015Data/Audio';

% Get Twin 2015 Relations
metamat=strcat('/home/stallone/Desktop/Twins2015Data/metadataTwins2015.xlsx'); 
[~,~,meta]=xlsread(metamat);
[row,col]=size(meta);
for m=2:row
    meta{m,1}=num2str(meta{m,1});
end

% Begin Matching for Twins 2015
twinRelId15={};
twinRelFr15={};
id=1;
fr=1;
for j=2:length(meta)
    RID=meta{j,1}; % get current RID  
%     if ismember(RID,listRIDvec)
%         for j=2:length(meta)
            if j==2
               twinRelId15{id,1} = RID;
               TTID=meta{j,6};
               for k=j+1:length(meta)
                  if meta{k,2}==TTID
                     twinRelId15{id,2}=meta{k,1};                      
                  end
               end 
               id=id+1;
            elseif ~ismember(RID,twinRelId15(:,1)) && ~ismember(RID,twinRelId15(:,2))% && j~=185 && j~=188 % ~strcmp(RID,meta{j-1,1}) &&%&& meta{j,2}~=meta{j-1,6}
               if strcmp(meta{j,9},'Identical') || strcmp(meta{j,9},'Identical Mirror')
                  twinRelId15{id,1} = RID;
                  TTID=meta{j,6};
                  for k=j+1:length(meta)
                      if meta{k,2}==TTID
                         twinRelId15{id,2}=meta{k,1};                      
                      end
                  end 
                  id=id+1;
               else 
                   msg=sprintf('RID %s is not an identical twin',RID);
                   disp(msg)
                   if j==5
                       twinRelFr15{fr,1} = RID;
                       TTID=meta{j,6};
                      for k=j+1:length(meta)
                          if meta{k,2}==TTID
                             twinRelFr15{fr,2}=meta{k,1};                      
                          end
                      end
                       fr=fr+1;
                   elseif strcmp(meta{j,9},'Fraternal') && ~ismember(RID,twinRelFr15(:,1)) && ~ismember(RID,twinRelFr15(:,2))% && j~=83
                      twinRelFr15{fr,1} = RID;
                      TTID=meta{j,6};
                      for k=j+1:length(meta)
                          if meta{k,2}==TTID
                             twinRelFr15{fr,2}=meta{k,1};                      
                          end
                      end
                       fr=fr+1;
                   else 
                      msg=sprintf('RID %s is not fraternal or identical or is already present in Frat',RID);
                      disp(msg)
                   end
               end
            end   
%         end
%     else
%         msg=sprintf('RID %s is not in metadata 2015',RID);
%         disp(msg) 
%     end
end

%% Twins 2016
clc 
clear all

% Get Twin 2016 Audio
listRID=dir(fullfile('/home/stallone/Desktop/Twins2016Data/Audio')); % list of RIDs  
ridPath = '/home/stallone/Desktop/Twins2016Data/Audio';

% Get Twin 2016 Relations
metamat=strcat('/home/stallone/Desktop/Twins2016Data/metadataTwins2016.xlsx'); 
[~,~,meta]=xlsread(metamat);
[row,col]=size(meta);
for m=2:row
    meta{m,1}=num2str(meta{m,1});
end

% Begin Matching for Twins 2015
twinRelId16={};
twinRelFr16={};
id=1;
fr=1;
for j=2:length(meta)
    RID=meta{j,1}; % get current RID  
%     if ismember(RID,listRIDvec)
%         for j=2:length(meta)
            if j==2
               twinRelFr16{fr,1} = RID;
               TTID=meta{j,6};
               for k=j+1:length(meta)
                  if meta{k,2}==TTID
                     twinRelFr16{fr,2}=meta{k,1};                      
                  end
               end 
               fr=fr+1;
            elseif ~ismember(RID,twinRelFr16(:,1)) && ~ismember(RID,twinRelFr16(:,2)) && ~strcmp(meta{j-1,1},RID) %j~=185 && j~=188 % ~strcmp(RID,meta{j-1,1}) &&%&& meta{j,2}~=meta{j-1,6}
               if strcmp(meta{j,9},'Fraternal') 
                  twinRelFr16{fr,1} = RID;
                  TTID=meta{j,6};
                  for k=j+1:length(meta)
                      if meta{k,2}==TTID
                         twinRelFr16{fr,2}=meta{k,1};                      
                      end
                  end 
                  fr=fr+1;
               else 
                   msg=sprintf('RID %s is not a fraternal twin',RID);
                   disp(msg)
                   if j==3
                       twinRelId16{id,1} = RID;
                       TTID=meta{j,6};
                      for k=j+1:length(meta)
                          if meta{k,2}==TTID
                             twinRelId16{id,2}=meta{k,1};                      
                          end
                      end
                       id=id+1;
                   elseif (strcmp(meta{j,9},'Identical') || strcmp(meta{j,9},'Identical Mirror')) && ~ismember(RID,twinRelId16(:,1)) && ~ismember(RID,twinRelId16(:,2)) && ~strcmp(meta{j-1,1},RID) %j~=83
                      twinRelId16{id,1} = RID;
                      TTID=meta{j,6};
                      for k=j+1:length(meta)
                          if meta{k,2}==TTID
                             twinRelId16{id,2}=meta{k,1};                      
                          end
                      end
                       id=id+1;
                   else 
                      msg=sprintf('RID %s is not fraternal or identical or is already present in Frat',RID);
                      disp(msg)
                   end
               end
            end   
%         end
%     else
%         msg=sprintf('RID %s is not in metadata 2016',RID);
%         disp(msg) 
%     end
end

%% Twins 2017 
clc 
clear all

% Get Twin 2017 Audio
listRID=dir(fullfile('/home/stallone/Desktop/Twins2017Data/Audio')); % list of RIDs  
ridPath = '/home/stallone/Desktop/Twins2017Data/Audio';

% Get Twin 2017 Relations
metamat=strcat('/home/stallone/Desktop/Twins2017Data/metadataTwins2017.xlsx'); 
[~,~,meta]=xlsread(metamat);
[row,col]=size(meta);
for m=2:row
    meta{m,1}=num2str(meta{m,1});
end

% Begin Matching for Twins 2015
twinRelId16={};
twinRelFr16={};
id=1;
fr=1;
for j=2:length(meta)
    RID=meta{j,1}; % get current RID  
%     if ismember(RID,listRIDvec)
%         for j=2:length(meta)
            if j==2
               twinRelFr16{fr,1} = RID;
               TTID=meta{j,6};
               for k=j+1:length(meta)
                  if meta{k,2}==TTID
                     twinRelFr16{fr,2}=meta{k,1};                      
                  end
               end 
               fr=fr+1;
            elseif ~ismember(RID,twinRelFr16(:,1)) && ~ismember(RID,twinRelFr16(:,2)) && ~strcmp(meta{j-1,1},RID) %j~=185 && j~=188 % ~strcmp(RID,meta{j-1,1}) &&%&& meta{j,2}~=meta{j-1,6}
               if strcmp(meta{j,9},'Fraternal') 
                  twinRelFr16{fr,1} = RID;
                  TTID=meta{j,6};
                  for k=j+1:length(meta)
                      if meta{k,2}==TTID
                         twinRelFr16{fr,2}=meta{k,1};                      
                      end
                  end 
                  fr=fr+1;
               else 
                   msg=sprintf('RID %s is not a fraternal twin',RID);
                   disp(msg)
                   if j==3
                       twinRelId16{id,1} = RID;
                       TTID=meta{j,6};
                      for k=j+1:length(meta)
                          if meta{k,2}==TTID
                             twinRelId16{id,2}=meta{k,1};                      
                          end
                      end
                       id=id+1;
                   elseif (strcmp(meta{j,9},'Identical') || strcmp(meta{j,9},'Identical Mirror')) && ~ismember(RID,twinRelId16(:,1)) && ~ismember(RID,twinRelId16(:,2)) && ~strcmp(meta{j-1,1},RID) %j~=83
                      twinRelId16{id,1} = RID;
                      TTID=meta{j,6};
                      for k=j+1:length(meta)
                          if meta{k,2}==TTID
                             twinRelId16{id,2}=meta{k,1};                      
                          end
                      end
                       id=id+1;
                   else 
                      msg=sprintf('RID %s is not fraternal or identical or is already present in Frat',RID);
                      disp(msg)
                   end
               end
            end   
%         end
%     else
%         msg=sprintf('RID %s is not in metadata 2016',RID);
%         disp(msg) 
%     end
end
