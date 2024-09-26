import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.lines as mlines  # Import Line2D to create custom legend handles
import os 

######################################################################################################################################################

RELATIVE = True #set calculation mode to relative

#fill in config here for Unit 1

Unit1_name = "RatlingGunPer100"

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

Unit2_name = "WarpfireThrowers250Per100"

Unit2_Units = 3                 #Number of models in unit 1 with that weapon profile

Unit2_Attacks = 3 * 3.5         #Number of Attacks

Unit2_Hit = 4                   #Hit Value

Unit2_Wound = 4                 #Wound Value

Unit2_Rend = 1                  #Rend Value

Unit2_Damage = 1                #Damage Value

Unit2_Crit = "2 Hits"           #Name of Crit Ability - Currently implemented: "2 Hits", "Auto Wound", "Mortal", "None"

Unit2_Champions = 0             #Number of Champions in that Unit

Unit2_Ability = None            #Ability - Not implemented yet

Unit2_Points = 150              #Points cost of 1 unit

Unit2_Squads = 1                #Number of Squads you want to compute for 

Unit2_Lastwpn = True            #Set this to True if this is the last weapon for this Unit you want to compute

Unit2_additionalWeapons = None  #Here a List of Arrays can be passed that should be calculated as weapons beyond the first 
                                #for the same unit like: [(...),(...)] for multi weapon calculation set the Lastwpn parameter False
                                #on all but the last weapon which is the last element in the list

########################################################################################################################################################################


# Weapon configurations: (Units, Attack, Hit, Wound, Rend, Damage, Crit, Champions, Ability, relative, Points, Squads, Lastwpn, additionalWeapons)
Unit1_config = (Unit1_Units, Unit1_Attacks, Unit1_Hit, Unit1_Wound, Unit1_Rend,
                Unit1_Damage, Unit1_Crit, Unit1_Champions, Unit1_Ability, RELATIVE,
                Unit1_Points, Unit1_Squads, Unit1_Lastwpn, Unit1_additionalWeapons)   
Unit2_config = (Unit2_Units, Unit2_Attacks, Unit2_Hit, Unit2_Wound, Unit2_Rend, Unit2_Damage,
                Unit2_Crit, Unit2_Champions, Unit2_Ability, RELATIVE, Unit2_Points, Unit2_Squads,
                Unit2_Lastwpn, Unit2_additionalWeapons)       

# Directory 
directory = f"{Unit1_name}_Vs_{Unit2_name}"
  
# Parent Directory path 
parent_dir = "/Users/michaeldoleschal/p4p/privat/WarhammerAOS/Plots"
  
# Path 
path = os.path.join(parent_dir, directory) 
  
# Create the directory 
os.mkdir(path) 

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


# Define the ranges for Save and Ward
save_range = np.arange(1, 7)  # Save from 1 to 6
ward_range = np.arange(1, 8)  # Ward from 1 to 7

# Create a meshgrid for Save and Ward values
save_grid, ward_grid = np.meshgrid(save_range, ward_range)

# Calculate the result values for both weapon configurations
result_grid_Unit1 = calc_with_config(Unit1_config, save_grid, ward_grid)
result_grid_Unit2 = calc_with_config(Unit2_config, save_grid, ward_grid)

# Find the intersections where the results are equal (or close)
intersection_points = np.isclose(result_grid_Unit1, result_grid_Unit2, atol=0.1)

# Create the 3D surface plot with intersections and dominance regions
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface for Ratling Gun
surf_ratling = ax.plot_surface(save_grid, ward_grid, result_grid_Unit1, cmap='Reds', alpha=0.6, label=Unit1_name)

# Plot the surface for Warlock Jezzails
surf_jezzails = ax.plot_surface(save_grid, ward_grid, result_grid_Unit2, cmap='Blues', alpha=0.6, label=Unit2_name)

# Highlight the intersection points
ax.scatter(save_grid[intersection_points], ward_grid[intersection_points], 
           result_grid_Unit1[intersection_points], color='green', s=50, label='Intersection')

# Add color bars for both surfaces
cbar = plt.colorbar(surf_ratling, ax=ax)
cbar.set_label(f'{Unit1_name} Result')
cbar_jezzails = plt.colorbar(surf_jezzails, ax=ax)
cbar_jezzails.set_label(f'{Unit2_name} Result')

# Invert the Save axis (X-axis)
ax.invert_xaxis()

# Set axis labels
ax.set_xlabel('Save (Inverted)')
ax.set_ylabel('Ward')
ax.set_zlabel('Result')

# Create custom legend handles
ratling_handle = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=10, label=Unit1_name)
jezzails_handle = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=10, label=Unit2_name)
intersection_handle = mlines.Line2D([], [], color='green', marker='o', linestyle='None', markersize=10, label='Intersection')

# Add the custom legend with the created handles
ax.legend(handles=[ratling_handle, jezzails_handle, intersection_handle], loc='upper left')

# Show the 3D surface plot with intersections
#plt.show()

plt.savefig(f"{path}/3D_Surface_{Unit1_name}_vs_{Unit2_name}.png", dpi=300)

# ======================
# Dominance plot (Optional 2D Projections)
# ======================

# Check which surface dominates in different regions
Unit1_dom = result_grid_Unit1 > result_grid_Unit2
Unit2_dom = result_grid_Unit2 > result_grid_Unit1

# Plot 2D comparison as a heatmap or contour to show which surface dominates
fig, ax = plt.subplots(figsize=(10, 7))

# Create a 2D dominance plot (Ratling vs Jezzails)
dominance_plot = np.where(Unit1_dom, 1, np.where(Unit2_dom, -1, 0))
c = ax.contourf(save_grid, ward_grid, dominance_plot, cmap='RdBu', alpha=0.5)

# Plot intersection lines in 2D
ax.contour(save_grid, ward_grid, result_grid_Unit1 - result_grid_Unit2, levels=[0], colors='green', linewidths=2)

# Add color bar to explain regions
cbar_dom = plt.colorbar(c)
cbar_dom.set_ticks([-1, 0, 1])
cbar_dom.set_ticklabels([f'{Unit2_name} Dominates', 'Equal', f'{Unit1_name} Dominates'])

# Set axis labels and title
ax.set_xlabel('Save')
ax.set_ylabel('Ward')
ax.set_title(f'Dominance of {Unit1_name} vs {Unit2_name}')

#plt.show()

plt.savefig(f"{path}/Dominance_{Unit1_name}_vs_{Unit2_name}.png", dpi=300)

# ==========================
# Create secondary 2D scatter + line plots for fixed Save and Ward values
# ==========================

# Generate color map for the scatter plots
def get_color_map(value):
    """Returns a color based on the provided value."""
    norm = plt.Normalize(vmin=1, vmax=6)  # Assuming Save values are between 1 and 6
    return cm.coolwarm(norm(value))

# Plot 1: Result as a function of Ward for fixed Save values (1, 2, 3, 4, 5, 6)
fig, ax = plt.subplots(figsize=(12, 8))

for save_value in range(1, 7):
    # Calculate results for both functions
    result_values_Unit1 = calc_with_config(Unit1_config,save_value, ward_range)
    result_values_Unit2 = calc_with_config(Unit2_config,save_value, ward_range)
    
    # Get color for the current save_value
    color = get_color_map(save_value)
    
    # Plot Ratling Gun (solid line)
    ax.scatter(ward_range, result_values_Unit1, color=color, alpha=0.7, 
               label=f"{Unit1_name} (Save = {save_value})", marker='o')
    ax.plot(ward_range, result_values_Unit1, color=color, alpha=0.5, linestyle='-', linewidth=1.5)  # Solid line
    # Annotate Ratling Gun
    ax.annotate(f'Save = {save_value}', 
                xy=(ward_range[-1], result_values_Unit1[-1]), 
                xytext=(ward_range[-1] + 0.5, result_values_Unit1[-1]), 
                arrowprops=dict(arrowstyle='->', color='black', lw=0.5), 
                fontsize=9, color=color)

    # Plot Warplock Jezzails (dashed line)
    ax.scatter(ward_range, result_values_Unit2, color=color, alpha=0.7, 
               label=f"{Unit2_name} (Save = {save_value})", marker='x')
    ax.plot(ward_range, result_values_Unit2, color=color, alpha=0.5, linestyle='--', linewidth=1.5)  # Dashed line
    # Annotate Warplock Jezzails
    ax.annotate(f'Save = {save_value}', 
                xy=(ward_range[-1], result_values_Unit2[-1]), 
                xytext=(ward_range[-1] + 0.5, result_values_Unit2[-1]), 
                arrowprops=dict(arrowstyle='->', color='black', lw=0.5), 
                fontsize=9, color=color)

ax.set_xlabel('Ward')
ax.set_ylabel('Result')
ax.set_title('Result vs Ward for Fixed Save Values')

# Add a legend with unique labels on the left side
unique_labels = set()
for line in ax.get_lines():
    if line.get_label() not in unique_labels:
        unique_labels.add(line.get_label())
ax.legend(loc='upper left')  # Adjust position to the left side

plt.tight_layout()
#plt.show()

plt.savefig(f"{path}/Ward_Scatter_{Unit1_name}_vs_{Unit2_name}.png", dpi=300)

# Plot 2: Result as a function of Save for fixed Ward values (1, 2, 3, 4, 5, 6, 7)
fig, ax = plt.subplots(figsize=(12, 8))

for ward_value in range(1, 8):
    # Calculate results for both functions
    result_values_Unit1 = calc_with_config(Unit1_config,save_range, ward_value)
    result_values_Unit2 = calc_with_config(Unit2_config,save_range, ward_value)
    
    # Get color for the current ward_value
    color = get_color_map(ward_value)
    
    # Plot Ratling Gun (solid line)
    ax.scatter(save_range, result_values_Unit1, color=color, alpha=0.7, 
               label=f"{Unit1_name} (Ward = {ward_value})", marker='o')
    ax.plot(save_range, result_values_Unit1, color=color, alpha=0.5, linestyle='-', linewidth=1.5)  # Solid line
    # Annotate Ratling Gun
    ax.annotate(f'Ward = {ward_value}', 
                xy=(save_range[-1], result_values_Unit1[-1]), 
                xytext=(save_range[-1] + 0.5, result_values_Unit1[-1]), 
                arrowprops=dict(arrowstyle='->', color='black', lw=0.5), 
                fontsize=9, color=color)

    # Plot Warplock Jezzails (dashed line)
    ax.scatter(save_range, result_values_Unit2, color=color, alpha=0.7, 
               label=f"{Unit2_name} (Ward = {ward_value})", marker='x')
    ax.plot(save_range, result_values_Unit2, color=color, alpha=0.5, linestyle='--', linewidth=1.5)  # Dashed line
    # Annotate Warplock Jezzails
    ax.annotate(f'Ward = {ward_value}', 
                xy=(save_range[-1], result_values_Unit2[-1]), 
                xytext=(save_range[-1] + 0.5, result_values_Unit2[-1]), 
                arrowprops=dict(arrowstyle='->', color='black', lw=0.5), 
                fontsize=9, color=color)

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
#plt.show()

plt.savefig(f"{path}/Save_Scatter_{Unit1_name}_vs_{Unit2_name}.png", dpi=300)

