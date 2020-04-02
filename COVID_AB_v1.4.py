
import numpy as np
import copy
import matplotlib.pyplot as plt
from cycler import cycler
import pandas
import time
import os

colors = ['#424242ff', '#b6311cff', '#DB5237ff', '#FF7253ff', '#FF9270ff', '#8DC3FFff', '#69b668ff']

class person:
  """
  Person within the agent-based disease model
  Each person is initiated with a specific amount of contacts, level of physical contact,
  age, medical history, disease resistance and more
  """
  def __init__(self, ID, n_contacts, infected, infective, day_inf, inf_fac, fate):
    """
    === NEEDS REWORK ===
    Update of documentation postponed to after revision to avoid double work
    """
    
    # person identification, used throughout the models - advised to count upwards from 0
    self.ID                         = ID
    # number of contacts for this person integer or float
    self.number_of_contacts         = n_contacts
    # infected / infective statusses - True/False
    self.infected                   = infected
    self.infective                  = infective 
    # keep track of the day of infection - integer
    self.day_of_infection           = day_inf
    # pre-defined period of infection duration - integer or float
    self.dead                       = False
    self.immune                     = False
    # pre-defined fate - 'non_symptomatic', 'symptomatic', 'severe', 'critical_recover' or 'critical_death'
    self.fate                       = fate
    # infectivity of this person - float (0..1]
    self.infection_factor           = inf_fac
    self.infectivity                = 0
    self.friends                    = []
  
  # TODO remove function definitions from person class and write as separate functions

  def die(self):
    """Mark person to be dead, no longer infective and immune to disease"""

    self.infective = False
    self.infected = False
    self.dead = True
    self.immune = True

  def immunize(self):
    """Mark person to be immune and no longer infective"""

    self.infective = False
    self.infected = False
    self.immune = True

  def infect(self, population_yesterday, today):
    """Updates infection status

    Infectivity of peers is determined with yesterday's status (=unchanging) to
    prevent higher chances of infection for the higher ID's

    Arguments
    ---------
    population_yesterday : array of person objects
        Array of persons with yesterday's status
    today : int or float 
        Indicates the days since the start of the simulation
    """

    contacts = self.friends

    # Infect others
    for ID in contacts:
      friend = population_yesterday[ID]
      if friend.infected and friend.infective:
        p_transmission = [friend.infectivity, 1 - friend.infectivity]

        transmission = np.random.choice([True, False], 1, p=p_transmission)
        
        if transmission:
          self.infected = True
          self.infective = True
          self.day_of_infection = today

  def progress_infection(self, today, infectivity_table):
    """Updates infectivity following the infectivity_table

    Arguments
    ---------
    today : int or float 
        Indicates the days since the start of the simulation
    infectivity_table : 2D-array of floats
        2D-array with each row the progression of infectivity for a specific fate, 
        columns the day since infection
    """

    relative_day = today - self.day_of_infection

    if self.fate == 'non_symptomatic':
      self.infectivity = infectivity_lookup(infectivity_table, 0, relative_day)
    elif self.fate == 'symptomatic':
      self.infectivity = infectivity_lookup(infectivity_table, 1, relative_day)
    elif self.fate == 'severe':
      self.infectivity = infectivity_lookup(infectivity_table, 2, relative_day)
    elif 'critical' in self.fate:
      self.infectivity = infectivity_lookup(infectivity_table, 3, relative_day)

  def update(self, population_today, population_yesterday, today, infectivity_table):
    """Updates persons status for today
    
    Checks status of person on the current day, and updates status accordingly

    Arguments
    ---------
    population_today : array of person objects
        Array of persons for today
    population_yesterday : array of person objects
        Array of persons with yesterday's status
    today : int or float 
        Indicates the days since the start of the simulation
    infectivity_table : 2D-array of floats
        2D-array with each row the progression of infectivity for a specific 
        fate, columns the day since infection
    """

    if self.immune:
      return
    # Check for infected state at self from yesterday
    if population_yesterday[self.ID].infected:
      self.progress_infection(today, infectivity_table)      
      # If infected, but no longer infective, end the disease 
      if self.infectivity <= 0.:
        if 'death' in self.fate:
          self.die()
          # print('Person {} died on the {}th day'.format(self.ID, day))
        else:
          self.immunize()
          # print('Person {} was cured on the {}th day'.format(self.ID, day))
    # Otherwise, try to catch infection from your friends
    else:
      self.infect(population_yesterday, today)

def infectivity_lookup(infectivity_table, fate, day):
  """Lookup infectivity for the referenced fate and day

  Arguments
  ---------
  infectivity_table : 2D-array of floats
      2D-array with each row the progression of infectivity for a specific 
      fate, columns the day since infection
  fate : int
      Identifier for fate, value 0, 1, 2, 3
  day : int 
      Days since infection

  Returns
  -------
  infectivity : float 
      Chance of infecting contacts
  """

  # Errors encountered with indices out of bounds, try-except to catch
  # these ans return the last value of the table
  try:
    infectivity = infectivity_table[fate, day]
  except IndexError as e:
    infectivity = infectivity_table[fate, -1]

  return infectivity

def fill_contact_matrix(contact_matrix, ID, n_contacts):
  """Adds entries to contact matrix for person ID and returns added friend IDs

  Arguments
  ---------
  contact_matrix : 2D-array of int8
      large 2D-array of 0's and 1's, size = population_size x population_size
  ID : int
      ID of current person
  n_contacts : int
      minimum number of contacts for each person

  Returns
  -------
  friends : array of int
      array of friend IDs
  """

  # Find current amount of friends of ID
  n_current_friends = np.sum(contact_matrix[ID,:])
  current_friends = np.flatnonzero(contact_matrix[ID,:])

  # If more friends required, pick new friends from everyone with a larger 
  # ID that is not already a friend
  if n_current_friends < n_contacts:
    available_contacts = np.where(contact_matrix[ID, ID + 1:] == 0)[0] + ID + 1
    n_new_friends = n_contacts - n_current_friends
    new_friends = np.random.choice(available_contacts, int(n_new_friends))

    # Place contacts both ways
    contact_matrix[ID, new_friends] = 1
    contact_matrix[new_friends, ID] = 1
    
    friends = np.append(current_friends, new_friends)
  else:
    friends = current_friends
  
  return friends

def construct_society(population_size, n_infected, n_contacts, fates, p_fates, 
                      figflag):
  """Sets up all variables for a new society and save contacts to disk
  
  Creates a new population, with connections between the agents. Generates 
  initial population for use in the time-simulation. Stores contact matrix in
  h5 format in the ./data folder for future reference.
  
  Arguments
  ---------
  population_size : int
      number of people to create
  n_infected : int
      number of infected at t = 0
  n_contacts : int
      minimum number of contacts for each person
  fates : array of str
      potential disease outcomes
  p_fates : array of float
      chance of each disease outcome [0..1]
  figflag : bool
      determines whether plots are made

  Returns
  -------
  population_initial : array of person objects
  """
  
  print('Constructing society\n---------------\n')

  ID_infected_0 = np.random.randint(0, population_size, n_infected)
  population_initial = []

  # Because of the way the contact matrix is made random, the script 
  # sometimes runs into a friendless person. Try again until everyone 
  # has friends, after too many tries we give up and quit
  for attempts in range(10):
    try:
      # contact matrix in int8 format to reduce size in RAM
      contact_matrix = np.zeros((population_size,population_size), dtype=np.int8)

      for ID in range(population_size):
        if ID in ID_infected_0:
          fate = np.random.choice(fates, 1, p=p_fates)[0]
          infected = True
          infective = True
          day_inf = 0
          infection_factor = 1

          population_initial.append(person(ID, n_contacts, infected, infective, day_inf, 
            infection_factor, fate))

          population_initial[ID].friends = fill_contact_matrix(contact_matrix, ID, n_contacts, 
            population_size)
        else:
          fate = np.random.choice(fates, 1, p=p_fates)[0]
          infected = False
          infective = False
          day_inf = 0
          infection_factor = 0

          population_initial.append(person(ID, n_contacts, infected, infective, day_inf, 
            infection_factor, fate))

          population_initial[ID].friends = fill_contact_matrix(contact_matrix, ID, n_contacts, 
            population_size)
      break
    except ValueError as e:
      if attempts < 9:
        print('Error while setting up the population, trying again')
        # time.sleep(0.1)
      else:
        print('Error while setting up the population, maximum tries (10) reached. Raising error.')
        raise

  if figflag:
    total_contacts = sum(contact_matrix)
    plot_contacts(total_contacts)

  store_data(contact_matrix)

  return population_initial

def plot_contacts(total_contacts):
  """Creates histogram of data
  
  Arguments
  ---------
  total_contacts : array of floats
      total contacts for each ID
  """

  fig1, ax1 = plt.subplots()
  ax1.hist(total_contacts, bins=range(1,int(total_contacts.max()+1)), edgecolor='k')

def store_data(contact_matrix):
  """Stores contact matrix to disk

  Stores contact matrix in folder ./data as contact_matrix.h5. 
  Overwrites existing file

  Arguments
  ---------
  contact_matrix : 2D-array of int8
      represents connections within society
  """

  print('Making a back-up of society\n---------------\n')

  if not os.path.exists('data'):
    os.mkdir('data')

  df = pandas.DataFrame(data=contact_matrix)
  # df.to_csv('contact_matrix.csv')
  df.to_hdf('data/contact_matrix.h5', 'matrix')

def load_society(population_size, n_contacts, fates, p_fates, figflag):
  """Load population matrix from an earlier society and read from it
  
  Reduces execution time when a good society has been made already
  
  === UNDER CONSTRUCTION ===
  """
  print('Loading pre-constructed society\n---------------\n')
  
  # TODO check the referenced contact matrix file, 
  # raise exception if it is not found or if it has the wrong size

  # set up network of people
  population_initial = []
  # create population contacts
  for attempts in range(10):
    try:

      contact_matrix = np.zeros((population_size,population_size), dtype=np.int8)

      # create persons, ID ranging from 0 to population size
      for ID in range(population_size):
        # set infectivity parameters differently for initially infected
        if ID in ID_infected_0:
          fate = np.random.choice(fates, 1, p=p_fates)[0]
          infected = True
          infective = True
          day_inf = 0
          infection_factor = 1
          # if 'death' in fate:
          #   print('This one should die')

          population_initial.append(person(ID, n_contacts, infected, 
            infective, day_inf, population_size, infection_factor, fate))
        else:
          fate = np.random.choice(fates, 1, p=p_fates)[0]
          infected = False
          infective = False
          day_inf = 0
          infection_factor = 0
          # if 'death' in fate:
          #   print('This one should die')

          population_initial.append(person(ID, n_contacts, infected, 
            infective, day_inf, population_size, infection_factor, fate))
      break
    # because of the way the contact matrix is made random, the script 
    # sometimes runs into a friendless person. Try again until everyone 
    # has friends, after too many tries we give up
    except ValueError as e:
      if attempts < 9:
        print('Error while setting up the population, trying again')
        # time.sleep(0.1)
      else:
        print('Error while setting up the population, maximum tries (10) reached. Raising error.')
        raise

  # TODO for reading a massive datafile: https://stackoverflow.com/questions/25962114/how-to-read-a-6-gb-csv-file-with-pandas
  # read contact matrix line by line, adding contact matrix data to each person in initial society

  total_contacts = sum(contact_matrix)

  if figflag:
    plot_contacts(total_contacts)

  return population_initial
  
def process_generated_data(population_over_time, n_days, figflag, colors):
  """Stores, processes and plots generated data

  [TODO] Stores raw data in folder ./data [/TODO]
  Extracts useful numbers from the full population over time and passes 
  these to the plotting function.

  Arguments
  ---------
  population_over_time : 2D-array of person objects
      population state at each timestep
  n_days : int
      duration of time-simulation
  figflag : bool
      determines whether results are plotted
  colors : array of strings
      expects HEX RGBA values for custom plot colors
  """

  # TODO write function for storing the passed through raw data
  # print('Storing generated data\n---------------\n')

  # store_generated_data()

  print('Extracting data from population\n---------------\n')

  n = {
    'susceptible' : np.zeros(n_days),
    'infected'    : np.zeros(n_days),
    'asym'        : np.zeros(n_days),
    'symp'        : np.zeros(n_days),
    'severe'      : np.zeros(n_days),
    'critical'    : np.zeros(n_days),
    'recovered'   : np.zeros(n_days),
    'dead'        : np.zeros(n_days),
  }

  for day in range(n_days):
    population = population_over_time[day]
    
    for p in population:
      if p.dead:
        n['dead'][day] += 1
      elif p.immune:
        n['recovered'][day] += 1
      elif p.infected:
        n['infected'][day] += 1
        if p.fate == 'non_symptomatic':
          n['asym'][day] += 1
        elif p.fate == 'symptomatic':
          n['symp'][day] += 1
        elif p.fate == 'severe':
          n['severe'][day] += 1
        elif 'critical' in p.fate:
          n['critical'][day] += 1
      else:
        n['susceptible'][day] += 1
    # print('Day {:#02}: {}\t{}\t{}\t{}'.format(day, n_susceptible[day], n_infected[day], n_recovered[day], n_dead[day]))
  
  if figflag:
    plot_extracted_data(n, n_days, colors)

def plot_extracted_data(n, n_days, colors):
  """Makes pretty plot of data in n
  
  Arguments
  ---------
  n : dict
      expected fields are 
        'dead'
        'critical'
        'severe'
        'symp'
        'asym'
        'recovered'
        'susceptible'
  n_days : int
      number of days of simulation
  colors : array of str
      expects HEX RGBA values for custom plot colors
  """

  print('Plotting extracted data\n---------------\n')

  if not os.path.exists('figures'):
    os.mkdir('figures')

  t = np.arange(n_days)
  width = 0.35

  fig2, ax2 = plt.subplots()
  # implements custom colors
  ax2.set_prop_cycle(cycler(color=colors))
  # TODO make the generation of legend tags and data plot automated
  #      (from the keys in the n dictionary?)
  s1 = ax2.stackplot(t, n['dead'], n['critical'], n['severe'], n['symp'], n['asym'], 
                     n['recovered'], n['susceptible'])
  ax2.legend(['$n_{dead}$', '$n_{critical}$', '$n_{severe}$', '$n_{symptomatic}$', 
              '$n_{asymptomatic}$', '$n_{recoverd}$', '$n_{susceptible}$'], 
             loc='upper left')
  ax2.set_xlabel('Time (days)')
  ax2.set_ylabel('People')
  ax2.set_title('Progression of infective disease. Model COVID_AB_v1.2')

  fig2.savefig('figures/{}_disease progression.png'.format(time.strftime('%y%m%d-%H%M%S')))

  # plt.show()

def run_time_simulation(population_initial, n_days, infectivity_table):
  """Runs agent based simulation for the indicated number of days
  
  Arguments
  ---------
  population_initial : array of person objects
      all persons with properties needed to start the simulation
  n_days : int
      length of the simulation
  infectivity_table : 2D-array of floats
      infectivity over the course of the disease

  Returns
  -------
  population_over_time : 2D-array of person objects
      Full course of population changes over time
  """

  print('Running time-simulation\n---------------\n')

  population_over_time = []
  population_over_time.append(population_initial)
  for day in range(1, n_days):
    # TODO  reduce memory usage. Saving all agents every day is excessive. 
    #       Save only the data we need:
    #         couple matrices for:
    #         - status (none, infected, dead, immune) -> int, changes over time
    #         - infectivity(?) -> float, changes over time

    # update what is population of yesterday and today
    population_yesterday = population_over_time[day-1]
    population_today = copy.deepcopy(population_yesterday)
    for p in population_today:
      p.update(population_today, population_yesterday, day, infectivity_table)

    population_over_time.append(population_today)

  print('Time-simulation concluded\n---------------\n')
  
  return population_over_time
  
def load_infectivity():
  """Loads infectivity table
  
  [TODO] Create file on disk and load from there [/TODO]

  Returns
  -------
  infectivity_table : 2D-array of floats
      infectivity over the disease progression
  """

  infectivity_table = np.array([
    [0.005, 0.005, 0.005, 0.005, 0.005, 0.010, 0.010, 0.005, 0.005, 0.005, 0.005, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.005, 0.010, 0.015, 0.015, 0.020, 0.025, 0.030, 0.025, 0.020, 0.015, 0.010, 0.005, 0.005, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.030, 0.030, 0.030, 0.030, 0.030, 0.025, 0.025, 0.020, 0.015, 0.010, 0.005, 0.005, 0.005, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.005, 0.010, 0.020, 0.025, 0.030, 0.035, 0.035, 0.035, 0.035, 0.035, 0.035, 0.030, 0.030, 0.025, 0.020, 0.015, 0.015, 0.015, 0.015, 0.010, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.000, 0.000]
  ])

  return infectivity_table

def agent_based_simulation( population_size=100000, n_infected=5, n_contacts=20, 
                            p_fates=[0.3, 0.56, 0.1, 0.03, 0.01], n_days=90, 
                            new_society=True, figflag=True ):
  """Most important function in the file, calls all other functions

  Arguments
  ---------
  population_size : int
  n_infected : int
      initially infected people
  n_contacts : int
      minimum number of contacts per agent
  p_fates : array of floats
      chance of each of the fates
  n_days : int
      how long to run the simulation
  new_society : bool
      choice to create a new society or try to load one from disk
  figflag : bool
      create plots or not
  
  """

  print('\nInitializing system\n---------------\n')

  fates = ['non_symptomatic', 'symptomatic', 'severe', 'critical_recover', 'critical_death']
  infectivity_table = load_infectivity()

  # TODO implement choice between constructing new society and using old network
  if new_society:
    population_initial = construct_society(population_size, n_infected, n_contacts, fates, p_fates, figflag)
  # else
    # load_society here
  
  print('Initial population set up\n---------------\n')

  population_over_time = run_time_simulation(population_initial, n_days, infectivity_table)
  process_generated_data(population_over_time, n_days, figflag, colors)

def main():
  agent_based_simulation()

if __name__ == '__main__':
  main()
