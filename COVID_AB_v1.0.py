
import numpy as np
import copy
import matplotlib.pyplot as plt
from cycler import cycler
import pandas

colors = ['#424242ff', '#b6311cff', '#276ba2ff', '#299727ff']

# define the class for each agent
class person:
  '''
  Person within the agent-based disease model
  Each person is initiated with a specific amount of contacts, level of physical contact,
  age, medical history, disease resistance and more
  '''
  # peron identificator, also used to index in contact matrix
  # ID = 0
  # # people have a different level of connectivity. Modelled with a heavy-tail distribution
  # number_of_contacts = 2
  # # more physical contact increases the chance at infecting contacted people
  # # level_of_physical_contact = 0.5

  # # age = 40
  # # medical history on a scale from 1 - 5. Used as risk factor
  # # medical_history = 1

  # # degree_of_infection = 0

  # infected = False

  # # quarantined = False

  # day_of_infection = 0

  # infection_duration = 0

  # # non_infective_period = 0

  # infective = False

  # # non_symptomatic_period = 0

  # # symptoms = False

  # # day_of_symptoms_onset = 0

  # # hospitalized = False

  # # hospital_duration = 0

  # # person fate is either 'immunize' or 'die'
  # fate = 'immunize'

  # dead = False

  # immune = False

  def __init__(self, ID, n_contacts, infected, infective, day_inf, non_inf_dur, inf_dur, 
    population_size):
    self.ID                         = ID
    self.number_of_contacts         = n_contacts
    self.infected                   = infected
    self.infective                  = infective 
    self.day_of_infection           = day_inf
    self.non_infective_period       = non_inf_dur
    self.infection_duration         = inf_dur
    self.dead                       = False
    self.immune                     = False
    self.fate                       = 'immunize'
    # self.level_of_physical_contact  = lvl_phys
    # self.age                        = age
    # self.medical_history            = med_hist
    # self.degree_of_infection        = 
    # self.non_symptomatic_period     = 
    # self.symptoms                   = 
    # self.day_of_symptoms_onset      = 
    # self.hospitalized               = 
    # self.hospital_duration          = 

    fill_contact_matrix(ID, n_contacts, population_size)

  def update(self, population_today, population_yesterday, today, contact_matrix):
    if self.dead:
      return
    # check for infectivity at self from yesterday
    if population_yesterday[self.ID].infective:
      # if past the duration of infection, either die or immunize
      if day - self.day_of_infection > self.infection_duration:
        if self.fate == 'die':
          self.die()
          # print('Person {} died on the {}th day'.format(self.ID, day))
        elif self.fate == 'immunize':
          self.immunize()
          # print('Person {} was cured on the {}th day'.format(self.ID, day))
      # otherwise, infect contacts
      else:
        self.infect(population_today, contact_matrix, today)
        # print('Person {} infected his friends on the {}th day'.format(self.ID, day))
    elif population_yesterday[self.ID].infected:
      if day - self.day_of_infection > self.non_infective_period:
        self.infective = True

  def die(self):
    '''
    set properties to those of a dead person
    no longer infective, immune to further infection
    '''
    self.infective = False
    self.infected = False
    self.dead = True
    self.immune = True

  def immunize(self):
    '''
    set properties to those of an immune person
    no longer infective or infected
    '''
    self.infective = False
    self.infected = False
    self.immune = True

  def infect(self, population_today, contact_matrix, today):
    '''
    infects friends according to the contact matrix
    '''
    contacts = np.flatnonzero(contact_matrix[:, self.ID])
    # contacts = contacts[0]
    # infect others
    for ID in contacts:
      infectee = population_today[ID]
      if not infectee.infected and not infectee.immune:
        infectee.infected = True
        infectee.day_of_infection = today

def fill_contact_matrix(ID, n_contacts, population_size):
  # find current amount of friends of ID
  # print(ID)
  n_current_friends = np.sum(contact_matrix[ID,:])
  
  if n_current_friends < n_contacts:
    # if more friends required, 
    # pick new friends from everyone with a larger ID that is not already a friend
    available_contacts = np.where(contact_matrix[ID,ID+1:] == 0)[0] + ID + 1
    n_new_friends = n_contacts - n_current_friends
    # print(int(n_current_friends), n_contacts, int(n_new_friends))
    # print(available_contacts)
    # print(type(available_contacts))
    new_friends = np.random.choice(available_contacts, int(n_new_friends))

    contact_matrix[ID,new_friends] = 1
    contact_matrix[new_friends,ID] = 1
    # print(contact_matrix)

print('\nInitializing system\n---------------\n')

population_size = 10000
ID_infected_0 = np.random.randint(0, population_size, 4)

n_contacts = 2
inf_dur = 15
non_inf_dur = 5

n_days = 5 * inf_dur

print('Constructing society\n---------------\n')
# construct contact matrix
contact_matrix = np.zeros((population_size,population_size))

population_over_time = []

# set up network of people
population_initial = []
# create population contacts
for attempts in range(10):
  try:
    for ID in range(population_size):
      # place the initially infected
      if ID in ID_infected_0:
        infected = True
        infective = True
        day_inf = 0
        population_initial.append(person(ID, n_contacts, infected, 
          infective, day_inf, non_inf_dur, inf_dur, population_size))
      else:
        infected = False
        infective = False
        day_inf = 0
        population_initial.append(person(ID, n_contacts, infected, 
          infective, day_inf, non_inf_dur, inf_dur, population_size))
    break
  # because of the way the contact matrix is made random, the script 
  # sometimes runs into a friendless person. Try again until everyone 
  # has friends
  except ValueError as e:
    if attempts < 9:
      print('Error while setting up the population, trying again')
    else:
      print('Error while setting up the population, maximum tries (10) reached. Raising error.')
      raise

# print(contact_matrix)
total_contacts = sum(contact_matrix)

fig1, ax1 = plt.subplots()
ax1.hist(total_contacts, bins=range(1,int(total_contacts.max()+1)), edgecolor='k')
# for p in population_initial:
#   contacts_overview.append(p.n_contacts)

print('Making a back-up of society\n---------------\n')

df = pandas.DataFrame(data=contact_matrix)
df.to_csv('contact_matrix.csv')

print('Initial population set up\n---------------\n')

population_over_time.append(population_initial)

# save network of people to disk

print('Running time-simulation\n---------------\n')

# daily disease update
for day in range(1, n_days):
  population_yesterday = population_over_time[day-1]
  population_today = copy.deepcopy(population_yesterday)
  
  # gather data on infectees etc. this day

  for p in population_today:
    # update(self, population_today, population_yesterday, day, contact_matrix):
    p.update(population_today, population_yesterday, day, contact_matrix)

  population_over_time.append(population_today)

print('Time-simulation concluded\n---------------\n')
print('Extracting data from population\n---------------\n')
n_susceptible = np.zeros(n_days)
n_infected    = np.zeros(n_days)
n_recovered   = np.zeros(n_days)
n_dead        = np.zeros(n_days)

# extract daily data
for day in range(n_days):
  population = population_over_time[day]
  
  # print('population length on day', day, ':', len(population))

  for p in population:
    if p.infected:
      n_infected[day] += 1
    elif p.immune:
      n_recovered[day] += 1
    elif p.dead:
      n_dead[day] += 1
    else:
      n_susceptible[day] += 1


  # print('Day {:#02}: {}\t{}\t{}\t{}'.format(day, n_susceptible[day], n_infected[day], n_recovered[day], n_dead[day]))

# save data on disease progression

print('Plotting extracted data\n---------------\n')
t = np.arange(n_days)
width = 0.35

# plot data of infectees etc. over time
fig2, ax2 = plt.subplots()

ax2.set_prop_cycle(cycler(color=colors))

# b1 = ax2.bar(t, n_dead, width)
# b2 = ax2.bar(t, n_infected, width, 
#   bottom = n_dead)
# b3 = ax2.bar(t, n_recovered, width, 
#   bottom = n_dead + n_infected)
# b4 = ax2.bar(t, n_susceptible, width, 
#   bottom = n_dead + n_infected + n_recovered)

s1 = ax2.stackplot(t, n_dead, n_infected, n_recovered, n_susceptible)
ax2.legend(['$n_{dead}$', '$n_{infected}$', '$n_{recoverd}$', '$n_{susceptible}$'])
ax2.set_xlabel('Time (days)')
ax2.set_ylabel('People')
ax2.set_title('Progression of infective disease. Model COVID_AB_v1')

plt.show()