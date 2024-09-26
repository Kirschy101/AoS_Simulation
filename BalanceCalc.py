import numpy as np
import matplotlib.pyplot as plt
import os

######################################################################################################################################################

RELATIVE = True #set calculation mode to relative

#fill in config here for Unit 1

Unit1_name = "RatlingGun"

Unit1_Units = 3                 #Number of models in unit 1 with that weapon profile

Unit1_Attacks = 3 * 3.5         #Number of Attacks

Unit1_Hit = 4                   #Hit Value

Unit1_Wound = 4                 #Wound Value

Unit1_Rend = 1                  #Rend Value

Unit1_Damage = 1                #Damage Value

Unit1_Crit = "2 Hits"           #Name of Crit Ability - Currently implemented: "2 Hits", "Auto Wound", "Mortal", "None"

Unit1_Champions = 0             #Number of Champions in that Unit

Unit1_Ability = None            #Ability - Not implemented yet

Unit1_Points = 150              #Points cost of 1 unit

Unit1_Squads = 1                #Number of Squads you want to compute for 

Unit1_Lastwpn = True            #Set this to True if this is the last weapon for this Unit you want to compute

Unit1_additionalWeapons = None  #Here a List of Arrays can be passed that should be calculated as weapons beyond the first 
                                #for the same unit like: [(...),(...)] for multi weapon calculation set the Lastwpn parameter False
                                #on all but the last weapon which is the last element in the list

########################################################################################################################################################################

#fill in config here for Unit 2

Unit2_name = "WarplockJezzails"

Unit2_Units = 3                 #Number of models in unit 1 with that weapon profile

Unit2_Attacks = 2        #Number of Attacks

Unit2_Hit = 4                  #Hit Value

Unit2_Wound = 3                #Wound Value

Unit2_Rend = 2                  #Rend Value

Unit2_Damage = 2                #Damage Value

Unit2_Crit = "Auto Wound"           #Name of Crit Ability - Currently implemented: "2 Hits", "Auto Wound", "Mortal", "None"

Unit2_Champions = 1             #Number of Champions in that Unit

Unit2_Ability = None            #Ability - Not implemented yet

Unit2_Points = 150              #Points cost of 1 unit

Unit2_Squads = 1                #Number of Squads you want to compute for 

Unit2_Lastwpn = True            #Set this to True if this is the last weapon for this Unit you want to compute

Unit2_additionalWeapons = None  #Here a List of Arrays can be passed that should be calculated as weapons beyond the first 
                                #for the same unit like: [(...),(...)] for multi weapon calculation set the Lastwpn parameter False
                                #on all but the last weapon which is the last element in the list

########################################################################################################################################################################

# Generalized function that computes result based on Save, Ward, and weapon parameters
def calc_distribution(Units, Attack, Hit, Wound, Rend, Damage, Save, Ward, Crit="None", Champions=0, Ability="None",relative=False, Points=1, Squads=1, Lastwpn = True, additionalWeapons=None):
    w = Squads * (Units * Attack + Champions)
    if Crit != "None":
        x = w * np.abs(Hit - 6) / 6
        if Crit == "Auto Wound":
            y = x * (np.abs(Wound - 6) + 1) / 6 + Attack / 6
        elif Crit == "Mortal":
            y = x * (np.abs(Wound - 6) + 1) / 6
            z = np.where((Save + Rend) > 6, y, y * (6 - (np.abs(Save + Rend - 6) + 1)) / 6 + Attack / 6)
        elif Crit == "2 Hits":
            x += 2 * Attack / 6
    else:
        x = w * (np.abs(Hit - 6) + 1) / 6
    if Crit != "Auto Wound":
        y = x * (np.abs(Wound - 6) + 1) / 6
    if Crit != "Mortal":
        z = np.where((Save + Rend) > 6, y, y * (6 - (np.abs(Save + Rend - 6) + 1)) / 6)

    a = np.where(Ward == 7, z, z * (6 - (np.abs(Ward - 6) + 1)) / 6)
    b = a * Damage

    if additionalWeapons!=None:
        for i in range(len(additionalWeapons)):
            b += calc_with_config(additionalWeapons[i], Save, Ward)

    if Lastwpn:
        if relative:
            b = b/Points * 100

    return b

# Function to handle the calculation for a given weapon configuration
def calc_with_config(config, Save, Ward):
    Units, Attack, Hit, Wound, Rend, Damage, Crit, Champions, Ability, relative, Points, Squads, Lastwpn, additionalWeapons = config
    return calc_distribution(Units, Attack, Hit, Wound, Rend, Damage, Save, Ward, Crit=Crit, Champions=Champions, Ability=Ability, relative=relative,Points=Points, Squads=Squads,Lastwpn=Lastwpn, additionalWeapons=additionalWeapons)


# Weapon configurations: (Units, Attack, Hit, Wound, Rend, Damage, Crit, Champions, Ability, relative, Points, Squads, Lastwpn, additionalWeapons)
Unit1_config = (Unit1_Units, Unit1_Attacks, Unit1_Hit, Unit1_Wound, Unit1_Rend,
                Unit1_Damage, Unit1_Crit, Unit1_Champions, Unit1_Ability, RELATIVE,
                Unit1_Points, Unit1_Squads, Unit1_Lastwpn, Unit1_additionalWeapons)   
Unit2_config = (Unit2_Units, Unit2_Attacks, Unit2_Hit, Unit2_Wound, Unit2_Rend, Unit2_Damage,
                Unit2_Crit, Unit2_Champions, Unit2_Ability, RELATIVE, Unit2_Points, Unit2_Squads,
                Unit2_Lastwpn, Unit2_additionalWeapons)       

if RELATIVE:
    relAddon = "Per100"
else:
    relAddon = ""

# Directory 
directory = f"{Unit1_name}{relAddon}_Vs_{Unit2_name}{relAddon}"
  
# Parent Directory path 
parent_dir = "/Users/michaeldoleschal/p4p/privat/AoS_Simulation/Plots/Efficiency"
  
# Path 
path = os.path.join(parent_dir, directory) 
  
try:
    os.mkdir(path)
except OSError as error:
    print(error)  



######################################################################################################################################################

# Plot 2: Result as a function of Save for fixed Ward values of 7 (no ward)
fig, ax = plt.subplots(figsize=(12, 8))

# Ward is irrelevant for comparing units so it is set to 7 which means no ward save
ward_value = 7

# Define the ranges for Save and Ward
save_range = np.arange(1, 7)  # Save from 1 to 6

# Calculate results for both functions
result_values_Unit1 = calc_with_config(Unit1_config,save_range, ward_value )
result_values_Unit2 = calc_with_config(Unit2_config,save_range, ward_value)

# Plot Ratling Gun (solid line)
ax.scatter(save_range, result_values_Unit1, color='red', alpha=0.7, 
            label=f"{Unit1_name}{relAddon} (Ward = {ward_value})", marker='o')
ax.plot(save_range, result_values_Unit1, color='red', alpha=0.5, linestyle='-', linewidth=1.5)  # Solid line
# Annotate Ratling Gun
ax.annotate(f'Ward = {ward_value}', 
            xy=(save_range[-1], result_values_Unit1[-1]), 
            xytext=(save_range[-1] + 0.5, result_values_Unit1[-1]), 
            arrowprops=dict(arrowstyle='->', color='black', lw=0.5), 
            fontsize=9, color='red')

# Plot Warplock Jezzails (dashed line)
ax.scatter(save_range, result_values_Unit2, color='blue', alpha=0.7, 
            label=f"{Unit2_name}{relAddon} (Ward = {ward_value})", marker='x')
ax.plot(save_range, result_values_Unit2, color='blue', alpha=0.5, linestyle='--', linewidth=1.5)  # Dashed line
# Annotate Warplock Jezzails
ax.annotate(f'Ward = {ward_value}', 
            xy=(save_range[-1], result_values_Unit2[-1]), 
            xytext=(save_range[-1] + 0.5, result_values_Unit2[-1]), 
            arrowprops=dict(arrowstyle='->', color='black', lw=0.5), 
            fontsize=9, color='blue')

ax.set_xlabel('Save')
ax.set_ylabel('Result')
ax.set_title('Result vs Save for Fixed Ward Values')

# Add a legend with unique labels on the left side
unique_labels = set()
for line in ax.get_lines():
    if line.get_label() not in unique_labels:
        unique_labels.add(line.get_label())
ax.legend(loc='upper left')  # Adjust position to the left side

plt.tight_layout()
plt.savefig(f"{path}/Save_Scatter_{Unit1_name}{relAddon}_vs_{Unit2_name}{relAddon}.png", dpi=300)

#######################################################################################################################################

# Plot 3: Result as a function of Save for fixed Ward values of 7
fig, ax = plt.subplots(figsize=(12, 8))

# Define the ranges for Save and Ward
save_range = np.arange(1, 7)  # Save from 1 to 6

# Calculate results for both functions
result_values_Unit1 = calc_with_config(Unit1_config,save_range, ward_value )
result_values_Unit2 = calc_with_config(Unit2_config,save_range, ward_value)

Relative_diff = result_values_Unit1/result_values_Unit2 

# Plot Ratling Gun (solid line)
ax.scatter(save_range, Relative_diff, color='red', alpha=0.7, 
            label=f"{Unit1_name} efficiency relative to {Unit2_name}", marker='o')
ax.plot(save_range, Relative_diff, color='red', alpha=0.5, linestyle='-', linewidth=1.5)  # Solid line

equal_line = np.zeros(6)+1

ax.plot(save_range, equal_line ,color='blue', alpha=0.5, linestyle='--', linewidth=1.5)  # Solid line
ax.annotate(f'Equal efficiency Line', 
            xy=(save_range[-1], equal_line[-1]), 
            xytext=(save_range[-1] + 0.5, equal_line[-1]), 
            arrowprops=dict(arrowstyle='->', color='black', lw=0.5), 
            fontsize=9, color='blue')

ax.set_xlabel('Save')
ax.set_ylabel('Relative Factor of efficiency')
ax.set_title('Efficiency Factor vs Save')
ax.set_yticks(np.arange(Relative_diff[0]*0.9, Relative_diff[-1]*1.1, 0.1), minor=True)

# Add a legend with unique labels on the left side
unique_labels = set()
for line in ax.get_lines():
    if line.get_label() not in unique_labels:
        unique_labels.add(line.get_label())
ax.legend(loc='upper left')  # Adjust position to the left side

plt.tight_layout()
plt.savefig(f"{path}/Efficiency_Scatter_{Unit1_name}_vs_{Unit2_name}.png", dpi=300)