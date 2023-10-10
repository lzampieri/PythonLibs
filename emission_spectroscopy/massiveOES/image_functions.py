# -*- coding: utf-8 -*-
"""
Created on Fri Aug 08 21:48:00 2014

@author: Petr
"""

import numpy
import math
import copy


try:
    from progressbar import progressbar
    pbar_imported = True
except ImportError: 
    pbar_imported = False

def peak_derivative(data_array):
    
    """
    Function takes array of data and peaks and valeys by finding second numerical
    derivative and returns list of arrays in this order [[peak position],
    [peak height], [valey position], [valey height]]
    """
    first_derivative = numpy.diff(data_array)
    result = [0]*len(data_array)
    
    for i in range(len(first_derivative)):
        first_derivative[i] = first_derivative[i]/numpy.abs(first_derivative[i])
    second_derivative = numpy.diff(first_derivative)
    for i in range(len(second_derivative)):
        second_derivative[i] = second_derivative[i]/-numpy.abs(second_derivative[i])
    result[0] = 0
    for i in range(len(second_derivative)):
        result[i+1] = second_derivative[i]
    result[len(data_array)-1] = 0

    peak_position = []
    peak_height = []
    valey_position = []
    valey_height = []

    for i in range(len(result)):
        if result[i] == 1:
            peak_position.append(i)
            peak_height.append(data_array[i])
        if result[i] == -1:
            valey_position.append(i)
            valey_height.append(data_array[i])
    
    ret_data = [peak_position, peak_height, valey_position, valey_height]
    return ret_data
    

def offset_removal(data_source, **kwargs):
    """
    takes 2D image in form [:,:] as first argument and returns similar image which has shifted columns 
    by offset given by y = 0.015*x where x is number of column in image and y is shift. Missing points
    are substituted by 0 and points which were moved out of image are lost.
    
    Function can be called with optional argument steepnes = arg, where arg than substitutes
    value in equation so it y = arg*x
    
    """
    
    source_new = copy.deepcopy(data_source)
    images,y,x = source_new.shape
    steepnes = kwargs.pop('steepnes', 0.015)
    if pbar_imported:
        p = progressbar.ProgressBar(x)
    for column in range(x):
        if pbar_imported:
            p.animate(column)
        offset = math.floor(column*steepnes)
        for row in range(y):
            if row+offset < y:
                source_new[:, row, column] = source_new[:, row  + offset, column]
            else: 
                source_new[:, row, column] = 0   
    return source_new
    
def fiber_locate(data_source):
    """
    
    """
    new_data = numpy.mean(data_source[:,:,:], axis = 0)
    dim_x = len(new_data[0])
    max_spectra_value = new_data.max()    
    
    drop = []
    
    for i in range (0,dim_x):
          drop.append(peak_derivative(new_data[:,i]))
    
    data = []
    data.append([])
    data.append([])
    data.append([])

    a = len(drop)
    for i in range(0,a):
        b = len(drop[i][0])
        max_spectra_value = max(drop[i][0])
        for j in range(1,b-1):
            try: 
                 
                 if (drop[i][1][j] - (drop[i][3][j] + drop[i][3][j-1])/2) > (max_spectra_value-1000)/50:
                    data[0].append(i)
                    data[1].append(drop[i][0][j])
                    data[2].append(drop[i][1][j])
            except IndexError:
                 pass
            
    max_value = max(data[2])
    max_index = data[2].index(max_value)
    max_position_y = data[1][max_index]
    column_of_interest_y = []
    for j in range(max(data[1])):
        column_of_interest_temp = []
        for i in range(len(data[0])):
            if data[0][i] == j:
                column_of_interest_temp.append(data[1][i])
        if len(column_of_interest_temp) > 2:
            column_of_interest_y.append(column_of_interest_temp)
    
    print len(column_of_interest_y)
            
    distances = []
    for j in range(len(column_of_interest_y)):
        for i in range(len(column_of_interest_y[j])-1):
            distances.append(column_of_interest_y[j][i+1] -column_of_interest_y[j][i])
    
    min_distance = min(distances)
    counter_base = 0
    value_base_sum = 0
    for i in range(len(distances)):
        if (min_distance*2) >= distances[i]:
            value_base_sum += distances[i]
            counter_base += 1
    
    value_base = float(value_base_sum)/counter_base
    
    values_higher_order = []
    counter_high_order = 0
    for i in range(len(distances)):
        if (min_distance*2) < distances[i]:
            multiplier = round(distances[i]/value_base,0)
            values_higher_order.append(distances[i]/float(multiplier))
            counter_high_order += 1
    value_high_order = sum(values_higher_order)/float(len(values_higher_order))
    
    step_value = (value_high_order*counter_high_order + value_base*counter_base)/(counter_base+counter_high_order)
    offset = max_position_y%step_value
    return [step_value, offset]
    
def find_offset(data_source):
    # returns steepnes and most populated row
    optimum_steepnes = []

    for k in range (0,30):
        steepnes = 0.001*k
        
        if k > 0:
            new_source = offset_removal(data_source, steepnes = steepnes)
        else:
            new_source = data_source
            
        new_data = numpy.mean(new_source[:,:,:], axis = 0)
        dim_x = len(new_data[0])
 #       max_spectra_value = new_data.max()    
        
        drop = []
        
        for i in range (0,dim_x):
              drop.append(peak_derivative(new_data[:,i]))
        
        data = []
        data.append([])
        data.append([])
        data.append([])
    
        a = len(drop)
        for i in range(0,a):
            b = len(drop[i][0])
            max_spectra_value =  max(drop[i][0])
            for j in range(1,b-1):
                try: 
                     
                     if (drop[i][1][j] - (drop[i][3][j] + drop[i][3][j-1])/2) > (max_spectra_value-1000)/50:
                        data[0].append(i)
                        data[1].append(drop[i][0][j])
                        data[2].append(drop[i][1][j])
                except IndexError:
                     pass
        max_row = [0, 0]
        row_sizes = []
        for i in range(max(data[1])):
            
            row_list = []
            for j in range(len(data[1])):
                if data[1][j] == i:
                    row_list.append(data[1][j])
            if (len(row_list)) > 0:
                row_sizes.append(len(row_list))
            if max_row[0] < len(row_list):
                max_row[0] =  len(row_list)
                max_row[1] = i
            
            
        row_sizes.sort(reverse=True)
        
        optimum_steepnes.append([numpy.average(row_sizes),k,max_row])
    
    return optimum_steepnes
    
def image_to_spectra(source_data):
    offset = find_offset(source_data)
    
    best_offset = [0,0,0]    
    
    for i in range(len(offset)):
        if offset[i][0] > best_offset[0]:
            best_offset = offset[i]
    
    source_after_offset = offset_removal(source_data, steepnes = best_offset[1]*0.01)
    fiber_step = fiber_locate(source_after_offset)
    
    offset_used = best_offset[2][1]%fiber_step[0]
    
    new_data = numpy.mean(source_after_offset[:,:,:], axis = 0)
    spectra_list = []
    dim = new_data.shape
    dim_y = int(dim[0]) 
    steps_y = int(dim_y/fiber_step[0])
    
    used_vals = [best_offset[1], offset_used, fiber_step[0]]
    
    for i in range(0,steps_y):
            if (((int(offset_used + i*fiber_step[0]) -3) > 0) and ((int(offset_used + i*fiber_step[0]) +3) < dim_y)):
                current_spectra = numpy.mean(new_data[int(offset_used + i*fiber_step[0])-3:int(offset_used + i*fiber_step[0])+3,:], axis = 0)
                spectra_list.append([int(offset_used + i*fiber_step[0]), current_spectra])
    return spectra_list, used_vals, offset
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
