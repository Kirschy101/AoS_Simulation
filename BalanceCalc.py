import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import os
import pandas as pd
import itertools

###################################################################################################################################################### 

# Generalized function that computes result based on Save, Ward, and weapon parameters
def calc_distribution(Name,Units, Attack, Hit, Wound, Rend, Damage, Crit, Champions, Ability, Points, Squads,Buffs, Range, Lastwpn, additionalWeapons, Save, Ward):
    abilities = list(Ability.split("-"))
    if 'AlloutAttack' in abilities:
        Hit = Hit - 1
    if 'Warpcog' in abilities:
        Wound = Wound - 4/6
        Rend = Rend + 1/6
    if 'Fleshmeld' in abilities: 
        Attack = Attack + 2/3
    w = Squads * (Units * Attack + Champions)

    if Crit != "NoCrit":
        x = w * np.abs(Hit - 6) / 6
        if Crit == "Auto Wound":
            y = (x * (np.abs(Wound - 6) + 1) / 6) + (w / 6)
        elif Crit == "Mortal":
            y = x * (np.abs(Wound - 6) + 1) / 6
            z = np.where((Save + Rend) > 6, y, (y * (6 - (np.abs(Save + Rend - 6) + 1)) / 6)) + (w / 6)
        elif Crit == "2 Hits":
            x += 2 * w / 6
        elif "ShockGauntlet" in abilities:
            x += 3.5 * w / 6

    
    else:
        x = w * (np.abs(Hit - 6) + 1) / 6
    if Crit != "Auto Wound":
        y = x * (np.abs(Wound - 6) + 1) / 6
    if Crit != "Mortal":
        z = np.where((Save + Rend) > 6, y, y * (6 - (np.abs(Save + Rend - 6) + 1)) / 6)

    a = np.where(Ward == 7, z, z * (6 - (np.abs(Ward - 6) + 1)) / 6)
    b = a * Damage

    if "DoomFlayer" in abilities:
        b += 1 # 1/3 * 1 damage + 1/3 * 2 Damage 

    if additionalWeapons != "NoWpn":
        additionalWeapons = additionalWeapons.replace("[","")
        additionalWeapons = additionalWeapons.replace("]","")
        li = list(additionalWeapons.split(";"))
        for j in range(len(li)):
            list_weapon = li[j].split(",")
            for k in range(len(list_weapon)):
                try:
                    list_weapon[k] = float(list_weapon[k])
                except:
                    pass
            #df_temp = pd.DataFrame(list_weapon,columns=['Name', 'Units', 'Attack','Hit','Wound','Rend','Damage','Crit','Champions','Ability','Points','Squads','Lastwpn','additionalWeapons'])
            b += calc_with_config_single(list_weapon, Save, Ward)

    if Lastwpn:
        if RELATIVE:
            b = b/Points * 100

    return b

# Function to handle the calculation for a given weapon configuration
def calc_with_config(df, Save, Ward):
    array = df.values.tolist()
    rows = len(array)
    firstrow = calc_distribution(*array[0][:], Save, Ward)
    retArray = firstrow
    for i in range(rows-1): 
        newrow = calc_distribution(*array[i+1][:], Save, Ward)
        retArray = np.vstack([retArray, newrow]) 
    return retArray

def calc_with_config_single(arglist, Save, Ward):
    firstrow = calc_distribution(*arglist, Save, Ward) 
    return firstrow


#implement the automated application of several buffs to all units

def addBuffRows(df):
    for index, row in df.iterrows():
        buffs = list(df.loc[index,'Buffs'].split('-'))
        if '' in buffs:
            buffs = buffs.remove('')
        if buffs:
            for buff in buffs:
                for L in range(len(buffs) + 1):
                    for subset in itertools.combinations(buffs, L):
                        if subset==False:
                            subset = ''
                        new_row = df.loc[index,:].copy()
                        new_row.loc['Buffs']=subset
                        buff_string = str(subset)
                        buff_string = buff_string.replace(',','')
                        buff_string = buff_string.replace('(','')
                        buff_string = buff_string.replace(')','')
                        buff_string = buff_string.replace(' ','')
                        buff_string = buff_string.replace("'","")

                        new_row.loc['Name'] += buff_string
                        df = df._append([new_row], ignore_index=True)
    
    for i in range(index+1):
        df = df.drop(index=i)
    df.index = df.index-(index+1)  # shifting index
              
    print(df)
    return df


def applybuff_row(df, row_index, buffname):
    if buffname=='AlloutAttack':
        df.loc[row_index,'Hit'] -= 1

    return df


def applybuffs(df):
    for index, row in df.iterrows():
        if df.loc[index,'Buffs']:
            for buff in df.loc[index,'Buffs']:
                df = applybuff_row(df, index, buff)
    
    return df
    

RELATIVE = True #set calculation mode to relative

df = pd.read_csv("/Users/michaeldoleschal/p4p/privat/AoS_Simulation/Plots/Efficiency/UnitsMeleeNoBuffs.csv",sep = ';') 

if RELATIVE:
    relAddon = "Per100"
else:
    relAddon = ""

df = addBuffRows(df)
df = applybuffs(df)

df.to_csv('out.csv', index=False)

# Directory 
#directory = f"{Unit1_name}{relAddon}_Vs_{Unit2_name}{relAddon}"

directory = "TEST"

# Parent Directory path 
parent_dir = "/Users/michaeldoleschal/p4p/privat/AoS_Simulation/Plots/Efficiency"
  
# Path 
path = os.path.join(parent_dir, directory) 
  
#try:
    #os.mkdir(path)
#except OSError as error:
    #print(error)   

######################################################################################################################################################

# Plot 2: Result as a function of Save for fixed Ward values of 7 (no ward)
fig, ax = plt.subplots(figsize=(16, 8))

# Ward is irrelevant for comparing units so it is set to 7 which means no ward save
ward_value = 7

# Define the ranges for Save and Ward
save_range = np.arange(1, 7)  # Save from 1 to 6

# Calculate results for both functions
result_values_array = calc_with_config(df,save_range, ward_value )
#result_values_Unit2 = calc_with_config(Unit2_config,save_range, ward_value)

rowsRet, colsRet = result_values_array.shape

colors = iter(cm.rainbow(np.linspace(0, 1, rowsRet)))

symbols = {'range3':'x','range10': 'o', 'range15': 's', 'range18': '^' }

for i in range(rowsRet):
    thiscolor = next(colors)
    result_values = result_values_array[i,:]

    # Plot Ratling Gun (solid line)
    if 10<=df.loc[i]['Range']<15:
        marker = symbols['range10']
    elif 15<=df.loc[i]['Range']<18:
        marker = symbols['range15']
    elif 18<=df.loc[i]['Range']:
        marker = symbols['range18']
    else:
        marker = symbols['range3']
    ax.scatter(save_range, result_values, color=thiscolor, alpha=0.7, 
                label=f"{df.loc[i]['Name']}{relAddon}", marker=marker)
    ax.plot(save_range, result_values, color=thiscolor, alpha=0.5, linestyle='-', linewidth=1.5)  # Solid line
    # Annotate Ratling Gun
    ax.annotate(f"{df.loc[i]['Name']}{relAddon}", 
                xy=(save_range[-1], result_values[-1]), 
                xytext=(save_range[-1] + 0.5, result_values[-1]), 
                arrowprops=dict(arrowstyle='->', color='black', lw=0.5), 
                fontsize=9, color=thiscolor)

ax.set_xlabel('Save')
ax.set_ylabel(f'Dmg{relAddon}')
ax.set_title('Expected Damage vs Save')

ax.set_xlim((0,8))

# Add a legend with unique labels on the left side
unique_labels = set()
for line in ax.get_lines():
    if line.get_label() not in unique_labels:
        unique_labels.add(line.get_label())
ax.legend(loc='upper left')  # Adjust position to the left side

#plt.tight_layout()
plt.show()
#plt.savefig(f"{path}/Save_Scatter_{Unit1_name}{relAddon}_vs_{Unit2_name}{relAddon}.png", dpi=300)

#######################################################################################################################################

# Plot 3: Result as a function of Save for fixed Ward values of 7
fig, ax = plt.subplots(figsize=(16, 8))

# Define the ranges for Save and Ward
save_range = np.arange(1, 7)  # Save from 1 to 6

rowsRet, colsRet = result_values_array.shape

colors = iter(cm.rainbow(np.linspace(0, 1, rowsRet)))

symbols = {'range3':'x','range10': 'o', 'range15': 's', 'range18': '^' }

Relative_diff_array = result_values_array[1:]/result_values_array[0,:]

for i in range(rowsRet-1):
    thiscolor = next(colors)

    # Plot Ratling Gun (solid line)
    if 10<=df.loc[i]['Range']<15:
        marker = symbols['range10']
    elif 15<=df.loc[i]['Range']<18:
        marker = symbols['range15']
    elif 18<=df.loc[i]['Range']:
        marker = symbols['range18']
    else:
        marker = symbols['range3']

    # Plot Ratling Gun (solid line)
    ax.scatter(save_range, Relative_diff_array[i,:], color=thiscolor, alpha=0.7, 
                label=f"{df.loc[i+1]['Name']} efficiency relative to {df.loc[0]['Name']}", marker=marker)
    ax.plot(save_range, Relative_diff_array[i,:], color=thiscolor, alpha=0.5, linestyle='-', linewidth=1.5)  # Solid line

    ax.annotate(f"{df.loc[i+1]['Name']}{relAddon}", 
                xy=(save_range[-1], Relative_diff_array[i,-1]), 
                xytext=(save_range[-1] + 0.5, Relative_diff_array[i,-1]), 
                arrowprops=dict(arrowstyle='->', color='black', lw=0.5), 
                fontsize=9, color=thiscolor)

    equal_line = np.zeros(6)+1

ax.plot(save_range, equal_line ,color='blue', alpha=0.5, linestyle='--', linewidth=1.5)  # Solid line
ax.annotate(f"{df.loc[0]['Name']} Equal efficiency Line", 
            xy=(save_range[-1], equal_line[-1]), 
            xytext=(save_range[-1] + 0.5, equal_line[-1]), 
            arrowprops=dict(arrowstyle='->', color='black', lw=0.5), 
            fontsize=9, color='blue')

ax.set_xlabel('Save')
ax.set_ylabel('Relative Factor of efficiency')
ax.set_title('Efficiency Factor vs Save')

ax.set_xlim((0,8))

#ax.set_yticks(np.arange(Relative_diff_array[0,0]*0.9, Relative_diff_array[-1,-1]*1.1, 0.1), minor=True)

# Add a legend with unique labels on the left side
unique_labels = set()
for line in ax.get_lines():
    if line.get_label() not in unique_labels:
        unique_labels.add(line.get_label())
ax.legend(loc='upper left')  # Adjust position to the left side

#plt.tight_layout()
plt.show()
#plt.savefig(f"{path}/Efficiency_Scatter_{Unit1_name}_vs_{Unit2_name}.png", dpi=300)