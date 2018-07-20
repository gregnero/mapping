#Author: Gregory Nero
#
#Purpose: plots word frequency from list as a barplot
#
#Date: July 20th, 2018

def word_frequency_barplot(mylist, title):
    
    '''
    Args:
    --------
    mylist: list of strings
    title: string

    Returns:
    --------
    A plot of the word frequencies in mylist

    Notes:
    --------
    Function requires matplotlib.pyplot

    Usage:
    --------
    word_frequency_barplot(list_of_animal_names, 'Animal Frequency')

    '''
    
    #Get all of the unique names from the input list
    list_of_unique_names = list(set(mylist))
    
    #Instantiate a dictionary for storing word:frequency pair
    frequency_dictionary = {}

    #Loop to fill the dictionary
    for unique_name in list_of_unique_names:
        
        frequency_counter = 0

        for name in mylist:
    
            if unique_name == name:
        
                frequency_counter += 1

            else:
                continue
    
        frequency_dictionary[unique_name] = frequency_counter


    #Strip the list information from the dictionary and prepare it for pass to plt.box()
    plot_names = sorted(frequency_dictionary, key=frequency_dictionary.get)
    plot_frequencies = sorted(frequency_dictionary.values())
    plot_names.reverse()
    plot_frequencies.reverse()


    #Matplotlib stuff
    fig, ax = plt.subplots()
    ax.set_ylabel('Word Frequency')
    ax.set_title(title)
    plt.bar(plot_names, plot_frequencies, width = 0.2) #width can be modified if needed
    plt.ion() #allows for multiple figures per "session"
    plt.show()

    #Inform that plotting was successful
    print(title + ': has been displayed')
