#Author: Gregory Nero
#
#Returns a barplot of word frequency in a list

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

    list_of_unique_names = list(set(mylist))
    
    frequency_dictionary = {}

    for unique_name in list_of_unique_names:
        
        frequency_counter = 0

        for name in mylist:
    
            if unique_name == name:
        
                frequency_counter += 1

            else:
                continue
    
        frequency_dictionary[unique_name] = frequency_counter

    plot_names = list(frequency_dictionary.keys())
    plot_frequencies = list(frequency_dictionary.values())

    fig, ax = plt.subplots()
    ax.set_ylabel('Word Frequency')
    ax.set_title(title)
    plt.bar(plot_names, plot_frequencies)
    plt.show()

