clc 
clear all

% Get Twins audio
listRID=dir(fullfile('/home/stallone/Desktop/Twins2014Data/Audio')); % list of RIDs
ridPath = '/home/stallone/Desktop/Twins2014Data/Audio';

% Get Twin Genders
demograph=strcat('/home/stallone/Desktop/Twins2014Data/demographicTwins2014.xlsx'); 
[~,~,demo]=xlsread(demograph);
[r,c]=size(demo);
for i=2:r 
   demo{i,1}=num2str(demo{i,1}); 
end
rids=demo(:,1);

% Get Twin Relationships
metamat=strcat('/home/stallone/Desktop/Twins2014Data/metadataTwins2014.xlsx');  
[~,~,meta] = xlsread(metamat);
[row,col]=size(meta);

% Making Master List with Twins 2014 Data
females = 0;
males = 0;
twinMstr={};
for i=3:length(listRID) % for all RIDs
    RID=listRID(i).name; % get current RID
    twinMstr{i-2,1} = RID;
    % Find if Female or Male
    [r,c]=find(strncmp(RID,demo,8)); 
     if (~isempty(r))
            r=r(1,1);
            if strcmp('Female',char(demo{r,3}))
                       twinMstr{i-2,2} = 'Female';
                       females = females+1; 
            else
                       twinMstr{i-2,2} = 'Male';
                       males = males+1;
            end
            listDATE=dir(fullfile(ridPath,RID)); % list of dates for current RID    
            for j=3:length(listDATE) % for all dates 
                DATE=listDATE(j).name; % get current date 
    %     if isdir(fullfile(ridPath14,RID))

                % Pathing to Subject and Rainbow File
                filename=strcat(RID,'_',DATE,'_subject.wav'); 
                filePath=fullfile(ridPath,RID,DATE,'Nuemann Mic',filename);
                filename2=strcat(RID,'_',DATE,'_rainbow.wav'); 
                filePath2=fullfile(ridPath,RID,DATE,'Nuemann Mic',filename2);

                % Check to see if exists, if so then put in twinMstr
                % First Day
                if DATE == '08022014'
                    if (exist(filePath,'file')>0) 
                        twinMstr{i-2,3} = filePath;
                    end

                    if (exist(filePath2,'file')>0) 
                        twinMstr{i-2,15} = filePath2;
                    end  
                % Second Day
                elseif DATE == '08032014'
                    if (exist(filePath,'file')>0) 
                        twinMstr{i-2,4} = filePath;
                    end

                    if (exist(filePath2,'file')>0) 
                        twinMstr{i-2,16} = filePath2;
                    end   
                end
            end
       

        % Count up how many files
        % for Subject
        if ~isempty(twinMstr{i-2,3}) && ~isempty(twinMstr{i-2,4})
            twinMstr{i-2,11} = 2; 
        elseif ~isempty(twinMstr{i-2,3}) || ~isempty(twinMstr{i-2,4})
            twinMstr{i-2,11} = 1; 
        else 
            twinMstr{i-2,11} = 0; 
        end
        % for Rainbow
        if ~isempty(twinMstr{i-2,15}) && ~isempty(twinMstr{i-2,16})
            twinMstr{i-2,23} = 2; 
        elseif ~isempty(twinMstr{i-2,15}) || ~isempty(twinMstr{i-2,16})
            twinMstr{i-2,23} = 1; 
        else 
            twinMstr{i-2,23} = 0; 
        end
     else
        msg=sprintf('RID %s is not in demographic',RID);
        disp(msg)
     end


end

% Display amount of M/F and number of files
msg = sprintf('There are %d females and %d males in 2014.', females, males);
disp(msg);

%% Master List for Twins 2015
listRID=dir(fullfile('/home/stallone/Desktop/Twins2015Data/Audio')); % list of RIDs  
ridPath = '/home/stallone/Desktop/Twins2015Data/Audio';

% Get Twin Gender
demograph=strcat('/home/stallone/Desktop/Twins2015Data/demographicTwins2015.xlsx'); 
[~,~,demo]=xlsread(demograph);
[r,c]=size(demo);
for i=2:r 
   demo{i,1}=num2str(demo{i,1}); 
end
rids=demo(:,1);
            
% Making Master List of Twins 2015 Data
females = 0;
males = 0;
% twinMstr={};
for i=3:length(listRID) % for all RIDs
    RID=listRID(i).name; % get current RID
    % Find if Female or Male
    [r,c]=find(strncmp(RID,demo,8)); 
     if (~isempty(r))  
        r=r(1,1);
         if strcmp('Female',char(demo{r,3}))
            females = females+1; 
         else
            males = males+1;
         end
         if ismember(RID,twinMstr(:,1))
             msg=sprintf('Repeat from 2014 to 2015 on RID %s',RID);
             disp(msg)
             for k=1:length(twinMstr)
                 if RID==twinMstr{k,1}
                     idx=k;
                 end 
             end
             listDATE=dir(fullfile(ridPath,RID)); % list of dates for current RID    
             for j=3:length(listDATE) % for all dates 
                 DATE=listDATE(j).name; % get current date 

                 % Pathing to Subject and Rainbow File
                 filename=strcat(RID,'_',DATE,'_021_001_ADIO_NEUM_AUD_0003.WAV'); 
                 filePath=fullfile(ridPath,RID,DATE,'Neumann Mic',filename);
                 filename2=strcat(RID,'_',DATE,'_021_001_ADIO_NEUM_AUD_0004.WAV'); 
                 filePath2=fullfile(ridPath,RID,DATE,'Neumann Mic',filename2);

                    % Check see if exists, if so then put in twinMstr
                    % First Day
                if DATE == '08082015'
                    if (exist(filePath,'file')>0) 
                        twinMstr{idx,5} = filePath;
                    end

                    if (exist(filePath2,'file')>0) 
                        twinMstr{idx,17} = filePath2;
                    end  
                % Second Day
                elseif DATE == '08092015'
                    if (exist(filePath,'file')>0) 
                        twinMstr{idx,6} = filePath;
                    end

                    if (exist(filePath2,'file')>0) 
                        twinMstr{idx,18} = filePath2;
                    end   
                end
            end
                
            % Count up how many files
            % for Subject
            if ~isempty(twinMstr{idx,5}) && ~isempty(twinMstr{idx,6})
                twinMstr{idx,12} = 2; 
            elseif ~isempty(twinMstr{idx,5}) || ~isempty(twinMstr{idx,6})
                twinMstr{idx,12} = 1; 
            else 
                twinMstr{idx,12} = 0; 
            end
            % for Rainbow
            if ~isempty(twinMstr{idx,17}) && ~isempty(twinMstr{idx,18})
                twinMstr{idx,24} = 2; 
            elseif ~isempty(twinMstr{idx,17}) || ~isempty(twinMstr{idx,18})
                twinMstr{idx,24} = 1; 
            else 
                twinMstr{idx,24} = 0; 
            end
           
            % RID is not in the list already
         else
             q=length(twinMstr)+1;
             twinMstr{q,1} = RID;
             if strcmp('Female',char(demo{r,3}))
                twinMstr{q,2} = 'Female';
             else
                twinMstr{q,2} = 'Male';
             end
             listDATE=dir(fullfile(ridPath,RID)); % list of dates for current RID    
             for j=3:length(listDATE) % for all dates 
                 DATE=listDATE(j).name; % get current date 
                 
                % Pathing to Subject and Rainbow File
                filename=strcat(RID,'_',DATE,'_021_001_ADIO_NEUM_AUD_0003.WAV'); 
                filePath=fullfile(ridPath,RID,DATE,'Neumann Mic',filename);
                filename2=strcat(RID,'_',DATE,'_021_001_ADIO_NEUM_AUD_0004.WAV'); 
                filePath2=fullfile(ridPath,RID,DATE,'Neumann Mic',filename2);

                % Check see if exists, if so then put in twinMstr
                % First Day
                if DATE == '08082015'
                    if (exist(filePath,'file')>0) 
                        twinMstr{q,5} = filePath;
                    end

                    if (exist(filePath2,'file')>0) 
                        twinMstr{q,17} = filePath2;
                    end  
                % Second Day
                elseif DATE == '08092015'
                    if (exist(filePath,'file')>0) 
                        twinMstr{q,6} = filePath;
                    end

                    if (exist(filePath2,'file')>0) 
                        twinMstr{q,18} = filePath2;
                    end   
                end
             end  

            % Count up how many files
            % for Subject
            if ~isempty(twinMstr{q,5}) && ~isempty(twinMstr{q,6})
                twinMstr{q,12} = 2; 
            elseif ~isempty(twinMstr{q,5}) || ~isempty(twinMstr{q,6})
                twinMstr{q,12} = 1; 
            else 
                twinMstr{q,12} = 0; 
            end
            % for Rainbow
            if ~isempty(twinMstr{q,17}) && ~isempty(twinMstr{q,18})
                twinMstr{q,24} = 2; 
            elseif ~isempty(twinMstr{q,17}) || ~isempty(twinMstr{q,18})
                twinMstr{q,24} = 1; 
            else 
                twinMstr{q,24} = 0; 
            end
         end
     
     else
        msg=sprintf('RID %s is not in demographic',RID);
        disp(msg)
     end

end

% Display amount of M/F and number of files
msg = sprintf('There are %d females and %d males in 2015.', females, males);
disp(msg);

%% Master List for Twins 2016
clc 
% clear all 

listRID=dir(fullfile('/home/stallone/Desktop/Twins2016Data/Audio')); % list of RIDs  
ridPath = '/home/stallone/Desktop/Twins2016Data/Audio';

% Get Twin Gender
demograph=strcat('/home/stallone/Desktop/Twins2016Data/metadataTwins2016.xlsx'); 
[~,~,demo]=xlsread(demograph);
[r,c]=size(demo);
for i=2:r 
   demo{i,1}=num2str(demo{i,1}); 
end
rids=demo(:,1);

% Making Master List of Twins 2016 Data
females = 0;
males = 0;
% twinMstr={};
for i=3:length(listRID) % for all RIDs
    RID=listRID(i).name; % get current RID
    % Find if Female or Male
    [r,c]=find(strncmp(RID,demo,8)); 
     if (~isempty(r))  
        r=r(1,1);
         if strcmp('Female',char(demo{r,10}))
            females = females+1; 
         else
            males = males+1;
         end
         if ismember(RID,twinMstr(:,1))
             msg=sprintf('Repeat from 2014/15 to 2016 on RID %s',RID);
             disp(msg)
             for k=1:length(twinMstr)
                 if RID==twinMstr{k,1}
                     idx=k;
                 end 
             end
             listDATE=dir(fullfile(ridPath,RID)); % list of dates for current RID    
             for j=3:length(listDATE) % for all dates 
                 DATE=listDATE(j).name; % get current date 

                 % Pathing to Subject and Rainbow File
                 filename=strcat(RID,'_',DATE,'_023_001_ADIO_PVI1_SUBJ.WAV'); 
                 filePath=fullfile(ridPath,RID,DATE,'Dynamic_Mic',filename);
                 filename2=strcat(RID,'_',DATE,'_023_001_ADIO_PVI1_RBOW.WAV'); 
                 filePath2=fullfile(ridPath,RID,DATE,'Dynamic_Mic',filename2);

                    % Check see if exists, if so then put in twinMstr
                    % First Day
                if DATE == '08062016'
                    if (exist(filePath,'file')>0) 
                        twinMstr{idx,7} = filePath;
                    end

                    if (exist(filePath2,'file')>0) 
                        twinMstr{idx,19} = filePath2;
                    end  
                % Second Day
                elseif DATE == '08072016'
                    if (exist(filePath,'file')>0) 
                        twinMstr{idx,8} = filePath;
                    end

                    if (exist(filePath2,'file')>0) 
                        twinMstr{idx,20} = filePath2;
                    end   
                end
            end
                
            % Count up how many files
            % for Subject
            if ~isempty(twinMstr{idx,7}) && ~isempty(twinMstr{idx,8})
                twinMstr{idx,13} = 2; 
            elseif ~isempty(twinMstr{idx,7}) || ~isempty(twinMstr{idx,8})
                twinMstr{idx,13} = 1; 
            else 
                twinMstr{idx,13} = 0; 
            end
            % for Rainbow
            if ~isempty(twinMstr{idx,19}) && ~isempty(twinMstr{idx,20})
                twinMstr{idx,25} = 2; 
            elseif ~isempty(twinMstr{idx,19}) || ~isempty(twinMstr{idx,20})
                twinMstr{idx,25} = 1; 
            else 
                twinMstr{idx,25} = 0; 
            end
           
            % RID is not in the list already
         else
             q=length(twinMstr)+1;
             twinMstr{q,1} = RID;
             if strcmp('Female',char(demo{r,10}))
                twinMstr{q,2} = 'Female';
             else
                twinMstr{q,2} = 'Male';
             end
             listDATE=dir(fullfile(ridPath,RID)); % list of dates for current RID    
             for j=3:length(listDATE) % for all dates 
                 DATE=listDATE(j).name; % get current date 
                 
                % Pathing to Subject and Rainbow File
                filename=strcat(RID,'_',DATE,'_023_001_ADIO_PVI1_SUBJ.WAV'); 
                filePath=fullfile(ridPath,RID,DATE,'Dynamic_Mic',filename);
                filename2=strcat(RID,'_',DATE,'_023_001_ADIO_PVI1_RBOW.WAV'); 
                filePath2=fullfile(ridPath,RID,DATE,'Dynamic_Mic',filename2);

                % Check see if exists, if so then put in twinMstr
                % First Day
                if DATE == '08062016'
                    if (exist(filePath,'file')>0) 
                        twinMstr{q,7} = filePath;
                    end

                    if (exist(filePath2,'file')>0) 
                        twinMstr{q,19} = filePath2;
                    end  
                % Second Day
                elseif DATE == '08072016'
                    if (exist(filePath,'file')>0) 
                        twinMstr{q,8} = filePath;
                    end

                    if (exist(filePath2,'file')>0) 
                        twinMstr{q,20} = filePath2;
                    end   
                end
             end  

            % Count up how many files
            % for Subject
            if ~isempty(twinMstr{q,7}) && ~isempty(twinMstr{q,8})
                twinMstr{q,13} = 2; 
            elseif ~isempty(twinMstr{q,7}) || ~isempty(twinMstr{q,8})
                twinMstr{q,13} = 1; 
            else 
                twinMstr{q,13} = 0; 
            end
            % for Rainbow
            if ~isempty(twinMstr{q,19}) && ~isempty(twinMstr{q,20})
                twinMstr{q,25} = 2; 
            elseif ~isempty(twinMstr{q,19}) || ~isempty(twinMstr{q,20})
                twinMstr{q,25} = 1; 
            else 
                twinMstr{q,25} = 0; 
            end
         end
     
     else
        msg=sprintf('RID %s is not in demographic',RID);
        disp(msg)
     end

end

% Display amount of M/F and number of files
msg = sprintf('There are %d females and %d males in 2016.', females, males);
disp(msg);

%% Master List for Twins 2017
clc 
clear all 

listRID=dir(fullfile('/home/stallone/Desktop/Twins2017Data/Audio')); % list of RIDs  
ridPath = '/home/stallone/Desktop/Twins2017Data/Audio';

% Get Twin Gender
demograph=strcat('/home/stallone/Desktop/Twins2017Data/metadataTwins2017.xlsx'); 
[~,~,demo]=xlsread(demograph);
[r,c]=size(demo);
for i=2:r 
   demo{i,1}=num2str(demo{i,1}); 
end
rids=demo(:,1);

% Begin making list for Twins 2017

females = 0;
males = 0;
% twinMstr={};
for i=3:length(listRID) % for all RIDs
    RID=listRID(i).name; % get current RID
    % Find if Female or Male
    [r,c]=find(strncmp(RID,demo,8)); 
     if (~isempty(r))  
        r=r(1,1);
         if strcmp('Female',char(demo{r,10}))
            females = females+1; 
         else
            males = males+1;
         end
         if ismember(RID,twinMstr(:,1))
             msg=sprintf('Repeat from 2014/15 to 2016 on RID %s',RID);
             disp(msg)
             for k=1:length(twinMstr)
                 if RID==twinMstr{k,1}
                     idx=k;
                 end 
             end
             listDATE=dir(fullfile(ridPath,RID)); % list of dates for current RID    
             for j=3:length(listDATE) % for all dates 
                 DATE=listDATE(j).name; % get current date 

                 % Pathing to Subject and Rainbow File
                 filename=strcat(RID,'_',DATE,'_023_001_ADIO_PVI1_SUBJ.WAV'); 
                 filePath=fullfile(ridPath,RID,DATE,'Dynamic_Mic',filename);
                 filename2=strcat(RID,'_',DATE,'_023_001_ADIO_PVI1_RBOW.WAV'); 
                 filePath2=fullfile(ridPath,RID,DATE,'Dynamic_Mic',filename2);

                    % Check see if exists, if so then put in twinMstr
                    % First Day
                if DATE == '08062016'
                    if (exist(filePath,'file')>0) 
                        twinMstr{idx,7} = filePath;
                    end

                    if (exist(filePath2,'file')>0) 
                        twinMstr{idx,19} = filePath2;
                    end  
                % Second Day
                elseif DATE == '08072016'
                    if (exist(filePath,'file')>0) 
                        twinMstr{idx,8} = filePath;
                    end

                    if (exist(filePath2,'file')>0) 
                        twinMstr{idx,20} = filePath2;
                    end   
                end
            end
                
            % Count up how many files
            % for Subject
            if ~isempty(twinMstr{idx,7}) && ~isempty(twinMstr{idx,8})
                twinMstr{idx,13} = 2; 
            elseif ~isempty(twinMstr{idx,7}) || ~isempty(twinMstr{idx,8})
                twinMstr{idx,13} = 1; 
            else 
                twinMstr{idx,13} = 0; 
            end
            % for Rainbow
            if ~isempty(twinMstr{idx,19}) && ~isempty(twinMstr{idx,20})
                twinMstr{idx,25} = 2; 
            elseif ~isempty(twinMstr{idx,19}) || ~isempty(twinMstr{idx,20})
                twinMstr{idx,25} = 1; 
            else 
                twinMstr{idx,25} = 0; 
            end
           
            % RID is not in the list already
         else
             q=length(twinMstr)+1;
             twinMstr{q,1} = RID;
             if strcmp('Female',char(demo{r,10}))
                twinMstr{q,2} = 'Female';
             else
                twinMstr{q,2} = 'Male';
             end
             listDATE=dir(fullfile(ridPath,RID)); % list of dates for current RID    
             for j=3:length(listDATE) % for all dates 
                 DATE=listDATE(j).name; % get current date 
                 
                % Pathing to Subject and Rainbow File
                filename=strcat(RID,'_',DATE,'_023_001_ADIO_PVI1_SUBJ.WAV'); 
                filePath=fullfile(ridPath,RID,DATE,'Dynamic_Mic',filename);
                filename2=strcat(RID,'_',DATE,'_023_001_ADIO_PVI1_RBOW.WAV'); 
                filePath2=fullfile(ridPath,RID,DATE,'Dynamic_Mic',filename2);

                % Check see if exists, if so then put in twinMstr
                % First Day
                if DATE == '08062016'
                    if (exist(filePath,'file')>0) 
                        twinMstr{q,7} = filePath;
                    end

                    if (exist(filePath2,'file')>0) 
                        twinMstr{q,19} = filePath2;
                    end  
                % Second Day
                elseif DATE == '08072016'
                    if (exist(filePath,'file')>0) 
                        twinMstr{q,8} = filePath;
                    end

                    if (exist(filePath2,'file')>0) 
                        twinMstr{q,20} = filePath2;
                    end   
                end
             end  

            % Count up how many files
            % for Subject
            if ~isempty(twinMstr{q,7}) && ~isempty(twinMstr{q,8})
                twinMstr{q,13} = 2; 
            elseif ~isempty(twinMstr{q,7}) || ~isempty(twinMstr{q,8})
                twinMstr{q,13} = 1; 
            else 
                twinMstr{q,13} = 0; 
            end
            % for Rainbow
            if ~isempty(twinMstr{q,19}) && ~isempty(twinMstr{q,20})
                twinMstr{q,25} = 2; 
            elseif ~isempty(twinMstr{q,19}) || ~isempty(twinMstr{q,20})
                twinMstr{q,25} = 1; 
            else 
                twinMstr{q,25} = 0; 
            end
         end
     
     else
        msg=sprintf('RID %s is not in demographic',RID);
        disp(msg)
     end

end

% Display amount of M/F and number of files
msg = sprintf('There are %d females and %d males in 2016.', females, males);
disp(msg);

