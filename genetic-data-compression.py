# -*- coding: utf-8 -*-
"""
Author: Madeline Hundley

This program uses a genetic algorithm in order to compress data.
For further information see the accompanying report.

Note: To change the level of logging statements, change the line at the beginning
of the main program.
- DEBUG is extremely verbose and shows detailed inner workings of the code.
- INFO shows the algorithm's steps.
"""

################################## IMPORTS ######################################

from bitarray import bitarray # used for codes and output data
import random # for probabilities and to generate input data
import logging # generates logging statements
import time # to test algorithm performance
from operator import attrgetter # for sorting
import math # for rounding, min, max, ceil, etc.
import numpy.random as nprandom # probabilities
import matplotlib.pyplot as plt # to graph stats
from datetime import datetime # naming stats
import csv # for saving stats

# Compare to other compression methods
from base64 import b64encode, b64decode
import ujson
import zlib
import lzma
import bz2

################### ALGORITHM CLASS AND FUNCTION DEFINITIONS ####################

class CodeDictionary(object):
    """
    Class for the members of the genetic population -- a dictionary mapping input
    digits to output codes

    Being variable-length, the codes must have the "prefix property", which
    requires that there is no whole codeword in the system that is a prefix of
    any other codeword in the system.

    Where necessary/applicable, attributes are of type 'string' to account for
    symbols that begin with '0'. This also makes the individual digits of the
    symbols easier to access.
    
    Instance attributes:
    symbols (list) - symbols to be mapped (str)
    codedict (dict) - keys: symbols to be mapped (str)
                      values: codes (bitarray)
    outdata (bitarray) - binary encoding of the input data
    outbyes (bytes) - the packed final product: codedict and outdata
    comp_ratio (float) - the compression ratio achieved

    Class attributes:
    pm - (float) the probability, Pm, between 0 and 1, of a mutation occuring in
                 the "genes" of the data
    pc - (float) the probability, Pc, between 0 and 1, of crossover occuring when
                 forming children from the "genes" of parent data
    """

    pm = 0.33 # the probability of mutation
    pc = 0.9 # the crossover probability    


    def __init__(self, codedict):
        """
        Initialize a new codedict
        """
        self.codedict = codedict
        self.symbols = list(codedict.keys())
        # Checking for this allows for dummy CodeDictionaries to be created
        if any(self.codedict):
            logging.debug("Initializing new Code Dictionary...")
            self.encode()
        return

    def __repr__(self):
        """
        Represent the CodeDictionary object
        """
        return str(self.codedict)

    def __str__(self):
        """
        Print the code dictionary
        """
        NUM_COLS = 4
        CODE_WIDTH = 8
        output = ""
        rows, rem = divmod(len(self.symbols), NUM_COLS)
        for i in range(rows):
            row = "|"
            for j in range(NUM_COLS):
                row += "{:>{w}}".format(self.symbols[(i*NUM_COLS)+j],
                                        w=MAX_SYM_LEN+1)
                row += ": "
                row += "{:<{w}}".format(self.codedict[
                    self.symbols[(i*NUM_COLS)+j]].to01(),w=CODE_WIDTH)
                row += "|"
            output += row
            output += "\n"

        if not rem:
            output += "      Compression Ratio: "
            output += "{:.7f}".format(self.comp_ratio)
            return output

        row = "|"
        for i in range(-rem,0):
            row += "{:>{w}}".format(self.symbols[i],w=MAX_SYM_LEN+1)
            row += ": "
            row += "{:<{w}}".format(self.codedict[self.symbols[i]].to01(),
                                    w=CODE_WIDTH)
            row += "|"
        output += row

        output += "\n      Compression Ratio: "
        output += "{:.7f}".format(self.comp_ratio)
        return output

    def __eq__(self, other):
        """
        Evaluates to True if this CodeDictionary and another have the same
        symbols and the same compression ratio
        """
        if self.symbols != other.symbols: return False
        if self.comp_ratio != other.comp_ratio: return False
        else: return True # they are functionally the same

    def encode (self):
        """
        Performs the encoding of the input data to get the output data; also
        calculates the compression ratio achieved
        Input: None
        Output: outdata (bitarray), comp_ratio (float)
        """
        logging.debug("Encoding {}...".format(INSTR))
        outdata = bitarray()

        sym_len = 1
        sym = ''
        for i in range(len(INSTR)):
            sym += INSTR[i]

            # When a symbol is found, encode and look for next symbol
            if sym in self.codedict:
                outdata += self.codedict[sym]
                sym_len = 1 # reset
                sym = '' # reset
                continue # move on

            # If checking for a symbol larger than possible, throw an error
            sym_len += 1
            if sym_len > MAX_SYM_LEN:
                raise Exception ("Couldn't match a symbol to input data")

        # If there are digits left at the end unmatched
        if sym: raise Exception ("Digits left unmatched in input data")

        logging.debug("Outdata is {}".format(outdata))
        self.outdata = outdata

        outbytes = pack(self)
        comp_ratio = len(INBYTES) / len(outbytes)
        logging.info("Compression ratio is " + str(comp_ratio))

        self.outbytes = outbytes
        self.comp_ratio = comp_ratio
        return


    def mutate(self):
        """
        Checks if a mutation occurs and performs the mutation if so
        """
        # Test if mutation will occur; if not, simply return
        if random.random() > CodeDictionary.pm: return

        logging.info("Attempting mutation...")

        # Randomly select a symbol and divide it into two
        # (Unless it is only 1 digit long, then do nothing)
        sym = random.sample(self.symbols,1)[0]
        if len(sym) == 1:
            logging.info("Mutation failed")
            return
        sym1 = sym[:len(sym)//2]
        sym2 = sym[len(sym)//2:]

        # Make sure they are not prefixes
        for curr_sym in self.symbols:
            if sym1+sym2 == curr_sym: continue # skip
            if sym1 == curr_sym[:len(sym1)] or sym2 == curr_sym[:len(sym2)]:
                logging.info("Mutation failed")
                return
        
        logging.info("Dividing {} into two new symbols: {} and {}".format(
            sym, sym1, sym2))
        logging.debug("Before:\n{}".format(self))

        # Remove old symbol and add new ones
        self.symbols.remove(sym)
        self.symbols.append(sym1)
        self.symbols.append(sym2)
        self.codedict.pop(sym)
        # Get new codes for the new symbols
        self.codedict = complete_codes(self.codedict, self.symbols)
        logging.debug("After:\n{}".format(self))

        # Recompute outdata, outbytes, and compression ratio
        self.encode()
        return



def single_encode(sym, data):
    """
    Function to find, remove, and demarcate all instances of a symbol in data
    Input: symbol (str), data (list[str])
    Output: data (list[str])
    """
    logging.debug("--- Running symbol {} against data ---".format(sym))
    i = 0
    while i < len(data):
        data_set = data[i]
        
        if len(data_set) < len(sym): # symbol too long
            i += 1 # move on to next digit set
            continue
        elif data_set == sym: # perfect match
            data.pop(i) # just get rid of this set of data
            continue # move on to next digit set (at this index)
        elif data_set[:len(sym)] == sym: # symbol at front
            data[i] = data_set[len(sym):] # save just remainder
            continue # check this digit set again
        elif data_set[-len(sym):] == sym: # symbol at end
            data[i] = data_set[:-len(sym)] # save just remainder
            continue # check this digit set again
        elif len(data_set) == len(sym) and data_set != sym:
            i += 1 # move on to next digit set
            continue 
        elif sym in data_set: # in middle of the digit set
            j = data_set.index(sym) # finds first occurrence
            # Save the data that came before symbol
            data[i] = data_set[:j]
            # Save the data that came after symbol
            data.insert(i+1, data_set[j+len(sym):])
            i += 1 # move to newly created next digit set
            continue # move on to next digit set

        i += 1 # if here, done with this digit set

    logging.debug("Data at end: {}".format(data))
    return data


def gen_codes(num_symbols):
    """
    Generates valid codes for the number of symbols
    Input: num_symbols (int)
    Output: codes (list[bitarray])
    """
    codes = [bitarray() for i in range(num_symbols)] # initialize
    
    def recursive_code(codes): # pass (portion of) codes
        if len(codes)//2 is 0: # recursion base case
            return codes
        else: # not done; divide and keep adding bits
            for b in range(len(codes)//2):
                codes[b] += bitarray('0') # "bottom" symbols get '0'
            for t in range(len(codes)//2, len(codes)):
                codes[t] += bitarray('1') # "top" symbols get '1'
            bot = recursive_code(codes[0:len(codes)//2])
            top = recursive_code(codes[len(codes)//2:len(codes)])
        return bot+top   

    recursive_code(codes)
    return codes


def get_rand_codedict():
    """
    Function to randomly generate a new codedict
    Input: None
    Output: A codedict (CodeDictionary)
    """
    logging.debug("Generating a new random code dictionary...")

    # Use empty, dummy kid and parent for complete_dict to process
    new_codedict = complete_dict(dict(), CodeDictionary(dict()))
    logging.debug("New codedict: {}".format(new_codedict))
    return CodeDictionary(new_codedict)
    

def get_init_pop():
    """
    Function to randomly generate an initial population of code dictionaries
    Input: None
    Output: a population (list) of codedicts (CodeDictionary)
    """
    init_pop = [] # list for the population of codedicts
    for i in range(POP_SIZE): # try to generate as many codedicts as we need
        try:
            new_codedict = get_rand_codedict()
        except Exception as e:
            logging.warning("complete_dict failed with Exception of " \
                            "type {}: {}".format(type(e),e))
            pass
        else:
            # check codedict is not a duplicate; save it in the population
            duplicate = False
            for cd in init_pop:
                if new_codedict == cd: # make use of custom function
                    duplicate = True
                    logging.debug("new codedict is a duplicate; not keeping")
                    break # stop looking
            else: init_pop.append(new_codedict)
            
    return init_pop


def locate_unmatched(kid_syms, unmatched):
    """
    Function finds the locations in INSTR of the provided unmatched digits
    Input: kid_syms (list[str]) and unmatched (str)
    Output: locations (set(int))
    """
    logging.debug("Finding {} with locate_unmatched".format(unmatched))
    data_locs = set(range(INDATA_NUM_DIGITS))

    i = 0
    while i < len(INSTR):
        for j in range(1,MAX_SYM_LEN+1):
            if INSTR[i:i+j] in kid_syms:
                data_locs = data_locs - set(range(i,i+j))
                i += j-1 # advance
                break # stop looking
        i += 1

    un_locs = set()

    for loc in data_locs:
        if INSTR[loc:loc+len(unmatched)] == unmatched:
            for i in range(0,len(unmatched)):
                if loc+i in data_locs: pass
                else: break
            else:
                logging.debug("Found {} at location {}".format(unmatched, loc))
                un_locs.add(loc)

    if un_locs:
        return un_locs
    else:
        logging.debug("Returning None")
        return None
    

def complete_dict(kid, parent):
    """
    Function to take the start of a code dictionary, find all necessary symbols
    to complete it (based on the parent's symbols) and compute the codes
    
    Can accept a dummy kid and dummy parent to funtionally generate a new,
    random codedict; in that case it uses gen_codes to compute the codes

    Input: incomplete kid codedict (dict[sym:code]) and parent codedict
           (CodeDictionary)
    Output: complete kid codedict (dict[sym:code])
    """
    logging.debug("Completing the code dictionary...")

    data = [INSTR] # list of strings
    for sym in kid: data = single_encode(sym, data[:])
    
    # Try to use parent's symbols to complete the dictionary

    for psym in parent.symbols:
        if not any(data): break # done looking
        logging.debug("Trying second parent's {}".format(psym))
        
        # If the other parent already gave this symbol
        if psym in kid:
            continue # move on

        # If symbol is a prefix to/of one already being used
        for ksym in kid:
            prefix = False
            if ksym == psym[0:len(ksym)]: # ksym is at front of psym
                prefix = True
                break # no need to keep checking
            elif psym == ksym[0:len(psym)]: # pysm is at front of ksym
                prefix = True
                break # no need to keep checking
        if prefix: continue # move on to next parent symbol

        # If symbols overlaps one already in use
        if find_overlaps(psym, list(kid.keys())):
            continue # move on to next parent symbol

        # Check if the symbol would be used by the data
        testdata = single_encode(psym, data.copy()) # process data with new symbol
        if testdata == data:
            continue # move on to the next symbol

        # Use this parent symbol
        data = testdata
        kid[psym] = parent.codedict[psym]
        logging.debug("Keeping parent symbol {}.".format(psym))

    # Fill in any gaps left after using parent's symbols

    kid_syms = list(kid.keys()) # track symbols that will be used
    conflicts = set() # symbols once removed that should not be reused
    overlaps = dict() # symbols not used because they overlap
    for s in kid_syms: overlaps[s] = set() # initialize

    # Compute initial dict of prefixes to/of current symbols
    logging.debug("Computing prefixes to/of current symbols...")
    prefixes = dict() # initialize dict for tracking
    for s in kid_syms: prefixes[s] = set() # initialize
    for p in prefixes:
        # Add symbols that prefix this one, and this symbol itself
        for i in range(len(p)): prefixes[p].add(p[:i+1])
        # Add symbols prefixed by this one
        if len(p) == MAX_SYM_LEN: continue # this check does not apply
        for i in range(10**(MAX_SYM_LEN-len(p))):
            prefixes[p].add(p+str(i))

    while any(data): # while there is still data to match

        # Enumerate all possible symbols for remaining data
        logging.debug("Enumerating all possible symbols for remaining data...")
        sym_choices = set()
        for data_set in data:
            length = min(len(data_set),MAX_SYM_LEN//2)
            for i in range(len(data_set)): # location
                for j in range(1, length+1): # length
                    sym_choices.add(data_set[i:i+j])

        # Remove conflicting symbols from possibilities
        logging.debug("Removing prefixes from choices...")
        # Never let prefixes be part of sym_choices
        for p in prefixes: sym_choices = sym_choices.difference(prefixes[p])
        if any(conflicts): # Remove conflicts from sym_choices
            logging.debug("Removing tracked conflicts: {}".format(conflicts))
            sym_choices.difference_update(conflicts) # remove all
        logging.debug("Removing tracked overlaps from choices...")
        for s in overlaps: # Remove overlaps from sym_choices
            sym_choices.difference_update(overlaps[s])

        # If there are valid choices; randomly select one of those
        if len(sym_choices) > 0:
            new_sym = random.sample(sym_choices,1)[0]
            logging.debug("Selected new_sym from choices.")
        else:
            new_sym = None
            logging.debug("There are no symbols left to choose from")
        

        # Do not keep any symbol that overlaps ones already chosen
        if new_sym:
            logging.debug("Checking new_sym {} for overlaps...".format(new_sym))
            new_sym_overlaps = find_overlaps(new_sym, kid_syms)
            if new_sym_overlaps:
                logging.debug("Keeping track of these; not using this symbol")
                for n in new_sym_overlaps:
                    overlaps[n].add(new_sym)
                continue # start over, find a different new_sym
            else: logging.debug("No overlap conflict with new_sym")

        # Do not keep any symbol that is contained within another
        if new_sym:
            logging.debug("Checking if new_sym is in any current symbol...")
            contains = False
            for k in kid_syms:
                if new_sym in k:
                    logging.debug("new_sym is inside {}".format(k))
                    overlaps[k].add(new_sym)
                    contains = True
            if contains:
                logging.debug("Not using this symbol")
                continue
                
        
        # If there are no valid choices; find matches for the remaining digits
        pre_sym = None
        if not new_sym: # Don't have a new symbol to try yet
            
            logging.debug("CONFLICT RESOLUTION")

            # Start at the front
            unmatched = data[0][:MAX_SYM_LEN]
            logging.debug("Trying to find digit(s) {} a symbol...".
                          format(unmatched))

            # Get first instance (or None)
            loc = sorted(locate_unmatched(kid_syms, unmatched))[0]
            logging.debug("Working on unmatched digit(s) at location {}".
                          format(loc))
            if loc is None: raise Exception("Failed to locate digit(s)")

            if loc is not 0:
                # Find the preceding symbol
                for i in range(1,MAX_SYM_LEN+1):
                    if INSTR[loc-i:loc] in kid_syms:
                        pre_sym = INSTR[loc-i:loc]
                        logging.debug("Symbol {} precedes unmatched".
                                      format(pre_sym))
                        break # done looking

        # If a symbol preceding the unmatched digit(s) was successfully located,
        # see if it can be removed and a new one formed
        cons = set() # track conflicts for processing below
        if pre_sym and not new_sym:
            
            # Try adding this hanging digit to the preceding symbol
            new_sym = (pre_sym+unmatched)[-MAX_SYM_LEN:]
            logging.debug("Old symbol {} may be removed, seeing about {} " \
                          "instead".format(pre_sym, new_sym))

            logging.debug("- Checking if new_sym is a prefix...")
            for p in prefixes:
                if p == pre_sym: continue # skip this match
                if new_sym in prefixes[p]:
                    logging.debug("  pre_sym+unmatched created a " \
                                  "prefix with {}".format(p))
                    new_sym = None # undo the selection
                    break # no need to keep checking


            # Check if the proposed new_sym is in conflicts
            if new_sym and new_sym in conflicts:
                logging.debug("- Reforming with preceding symbol is a conflict")
                new_sym = None # undo this selection


            # See if new_sym overlaps any current symbols
            if new_sym:
                logging.debug("- Checking new_sym {} for overlaps...".
                              format(new_sym))
                new_sym_overlaps = find_overlaps(new_sym, kid_syms)
                if new_sym_overlaps == {pre_sym}:
                    logging.debug("- No overlap conflict with new_sym")
                    pass
                elif new_sym_overlaps:
                    logging.debug("- Not going to use this symbol.")
                    new_sym = None # go to next step
            

            # Attempting all updates; seeing if new_sym works
            if new_sym:
                logging.debug("- Removing current symbol {}".format(pre_sym))
                kid_syms.remove(pre_sym) # Remove from the chosen symbols
                prefixes.pop(pre_sym) # Remove it from prefix tracking
                overlaps.pop(pre_sym) # Remove it from overlap tracking
                try: kid.pop(pre_sym) # Remove it from kid if present
                except: pass # If not, that's fine

                logging.debug("- Resetting the data...")
                data = [INSTR] # Start with fresh data and reprocess
                for sym in kid_syms: data = single_encode(sym, data[:])
            
                # Double check the new symbol is needed
                testdata = data.copy()
                data = single_encode(new_sym, data.copy())
                if testdata == data: # the new symbol is not needed
                    logging.debug("- New_sym {} wasn't found...".format(new_sym))
                    new_sym = None
                else:
                    logging.debug("- Keeping symbol {}".format(new_sym))
                    kid_syms.append(new_sym)
                    logging.debug("- Current symbols: {}".format(kid_syms))
                    # Update overlap tracking
                    logging.debug("- Updating overlap tracking...")
                    overlaps[new_sym] = set() # initialize new empty set
                    overlaps[new_sym].add(new_sym[-1]) # easy kill
                    # Update prefix tracking
                    logging.debug("- Updating prefixes to/of current symbols...")
                    # Add symbols that prefix this one, and this symbol itself
                    prefixes[new_sym] = set() # initialize new empty set
                    for i in range(len(new_sym)):
                        prefixes[new_sym].add(new_sym[:i+1])
                    # Add symbols prefixed by this one
                    if len(new_sym) == MAX_SYM_LEN:
                        continue # this check does not apply
                    for i in range(10**(MAX_SYM_LEN-len(new_sym))):
                        prefixes[new_sym].add(new_sym+str(i))
                    continue # done with new symbol, go back to top


        # If that was not successful, see if the unmatched digit(s)
        # can be added to the symbol that follows them
        post_sym = None
        if not new_sym:
            logging.debug("No preceding symbol or there was a conflict.")

            # Find the trailing symbol
            for i in range(1,MAX_SYM_LEN+1):
                if INSTR[loc+len(unmatched):loc+len(unmatched)+i] in kid_syms:
                    post_sym = INSTR[loc+len(unmatched):loc+len(unmatched)+i]
                    logging.debug("Found symbol {} following unmatched " \
                                  "digit(s)".format(post_sym))
                    break # done looking


        # If a symbol following the unmatched digit(s) was successfully located,
        # see if it can be removed and a new one formed
        if post_sym and not new_sym:

            # Try adding the unmatched digit(s) to the trailing symbol
            new_sym = (unmatched+post_sym)[:MAX_SYM_LEN]
            logging.debug("Old symbol {} may be removed, seeing about {} " \
                          "instead".format(post_sym, new_sym))

            logging.debug("~ Checking if new_sym is a prefix...")
            for p in prefixes:
                if new_sym in prefixes[p]:
                    logging.debug("post_sym+unmatched created a " \
                                  "prefix with {}".format(p))
                    logging.debug("Getting rid of that symbol instead...")
                    cons.add(p)
                    break # no need to keep checking

            # Check if we're at the front so the old symbol won't be readded
            if loc == 0:
                logging.debug("We're at the front so don't let the old symbol " \
                              "get re-added.")
                logging.debug("Adding to conflicts.")
                conflicts.add(post_sym)

            # Check if the proposed new_sym is in conflicts; this indicates add-
            # ing these digits to the front and to the back has failed
            # One reason for this is a "yo-yo" e.g. given '44083848', '44' and
            # '48' could keep gaining and losing '0838' in a loop
            if new_sym and new_sym in conflicts:
                logging.debug("Reforming with trailing symbol is a conflict.")
                new_sym = None # undo this selection
            
            if new_sym:
                logging.debug("~ Checking new_sym {} for overlaps...".
                              format(new_sym))
                new_sym_overlaps = find_overlaps(new_sym, kid_syms)
                if new_sym_overlaps == {post_sym}:
                    logging.debug("~ No overlap conflict with new_sym")
                    pass
                elif new_sym_overlaps:
                    logging.debug("~ Not going to use this symbol.")
                    new_sym = None # go to next step

            # Attempting all updates; seeing if new_sym works
            if new_sym:
                logging.debug("~ Making update; evaluating if it works")
                logging.debug("~ Removing current symbol {}".format(post_sym))
                kid_syms.remove(post_sym) # Remove from the chosen symbols
                prefixes.pop(post_sym) # Remove it from prefix tracking
                overlaps.pop(post_sym) # Remove it from overlap tracking
                try: kid.pop(post_sym) # Remove it from kid if present
                except: pass # If not, that's fine
                if any(cons):
                    for c in cons:
                        logging.debug("~ Removing current symbol {}".format(c))
                        try: kid_syms.remove(c)
                        except: pass
                        prefixes.pop(c)
                        overlaps.pop(c)
                        try: kid.pop(c)
                        except: pass

                logging.debug("~ Resetting the data...")
                data = [INSTR] # Start with fresh data and reprocess
                for sym in kid_syms: data = single_encode(sym, data[:])
            
                # Double check the new symbol is needed
                testdata = data.copy()
                data = single_encode(new_sym, data.copy())
                if testdata == data: # the new symbol is not needed
                    logging.debug("~ New_sym {} didn't work...".format(new_sym))
                    new_sym = None
                else:
                    logging.debug("~ Keeping symbol {}".format(new_sym))
                    kid_syms.append(new_sym)
                    logging.debug("~ Recording old symbol {} as a conflict".
                                  format(post_sym))
                    conflicts.add(post_sym)
                    logging.debug("~ Current symbols: {}".format(kid_syms))
                    # Update overlap tracking
                    logging.debug("~ Updating overlap tracking...")
                    overlaps[new_sym] = set() # initialize new empty set
                    overlaps[new_sym].add(new_sym[-1]) # easy kill
                    # Update prefix tracking
                    logging.debug("~ Updating prefixes to/of current symbols...")
                    # Add symbols that prefix this one, and this symbol itself
                    prefixes[new_sym] = set() # initialize new empty set
                    for i in range(len(new_sym)):
                        prefixes[new_sym].add(new_sym[:i+1])
                    # Add symbols prefixed by this one
                    if len(new_sym) == MAX_SYM_LEN:
                        continue # this check does not apply
                    for i in range(10**(MAX_SYM_LEN-len(new_sym))):
                        prefixes[new_sym].add(new_sym+str(i))
                    continue # done with new symbol, go back to top



        # If neither adding to the front or to the back was successful,
        # remove both to try again
        if not new_sym and pre_sym and post_sym:
            logging.debug("Both preceding and following symbols failed to " \
                          "form a good new symbol.")

            # Remove both and track so they aren't re-used
            conflicts.add(pre_sym)
            try: kid_syms.remove(pre_sym)
            except: pass
            try: prefixes.pop(pre_sym)
            except: pass
            try: overlaps.pop(pre_sym)
            except: pass
            try: kid.pop(pre_sym)
            except: pass

            conflicts.add(post_sym)
            try: kid_syms.remove(post_sym)
            except: pass
            try: prefixes.pop(post_sym)
            except: pass
            try: overlaps.pop(post_sym)
            except: pass
            try: kid.pop(post_sym)
            except: pass

            logging.debug("Resetting the data...")
            data = [INSTR] # Start with fresh data and reprocess
            for sym in kid_syms: data = single_encode(sym, data[:])

            continue
        
        
        # If unsuccessful, cannot proceed
        if not new_sym:
            raise Exception("Could not find a complete set of symbols.")

        # By this point, a new_sym has been chosen
        logging.debug("Trying new symbol {}".format(new_sym))

        # Remove any symbol(s) identified as needing removing
        if any(cons):
            logging.debug("Removing current symbol(s) {}".format(cons))
            conflicts.update(cons) # Track these so they aren't reused later
            for c in cons:
                try: kid_syms.remove(c) # Remove from the chosen symbols
                except: pass
                try: prefixes.pop(c) # Remove it from prefix tracking
                except: pass
                try: kid.pop(c) # Remove it from kid if present
                except: pass # If not, that's fine

            logging.debug("Resetting the data...")
            data = [INSTR] # Start with fresh data and reprocess
            for sym in kid_syms: data = single_encode(sym, data[:])

        # Double check the new symbol is needed
        testdata = data.copy()
        data = single_encode(new_sym, data.copy())
        if testdata == data: # the new symbol is not needed
            logging.debug("Didn't need to keep {}. Moving on...".format(new_sym))
            continue # go back and re-try without keeping this symbol

        # Keep the new symbol
        kid_syms.append(new_sym)
        logging.debug("Keeping symbol {}".format(new_sym))
        logging.debug("Current symbols: {}".format(kid_syms))
        # Update overlap tracking
        logging.debug("Updating overlap tracking...")
        overlaps[new_sym] = set() # initialize new empty set
        overlaps[new_sym].add(new_sym[-1]) # easy kill
        # Update prefix tracking
        logging.debug("Updating prefixes to/of current symbols...")
        # Add symbols that prefix this one, and this symbol itself
        prefixes[new_sym] = set() # initialize new empty set
        for i in range(len(new_sym)): prefixes[new_sym].add(new_sym[:i+1])
        # Add symbols prefixed by this one
        if len(new_sym) == MAX_SYM_LEN: continue # this check does not apply
        for i in range(10**(MAX_SYM_LEN-len(new_sym))):
            prefixes[new_sym].add(new_sym+str(i))


    logging.debug("Chose symbols: {}".format(kid_syms))

    return complete_codes(kid, kid_syms)


def find_overlaps(sym, list_syms):
    """
    Function compares the provided symbol against the others provided in the
    list; if it finds an overlapping symbol that exists in the indata, it
    returns it, otherwise returns 'None'
    Input: sym (str) and list_syms (list[str])
    Output: overlaps (set(str)) or None
    """
    overlaps = set()
    for k in list_syms:
        for i in range(1,len(sym)+1):
            if k[-i:] == sym[:i]: # k<->sym
                if k+sym[i:] in INSTR:
                    logging.debug("  Sym overlaps {} as {} in the data".
                                  format(k,k+sym[i:]))
                    overlaps.add(k)
            if sym[-i:] == k[:i]: # sym<->k
                if sym+k[i:] in INSTR:
                    logging.debug("  Sym overlaps {} as {} in the data".
                                  format(k,sym+k[i:]))
                    overlaps.add(k)

    if overlaps: return overlaps
    else: return None


def complete_codes (kid, kid_syms):
    """
    Function to take a complete set of symbols but incomplete codedict and
    complete the set of codes
    Input: kid (dict[sym:code]) and kid_syms (list)
    Output: codedict (dict[sym:code])
    """
    new_kid = kid.copy()
    codes = gen_codes(len(kid_syms))
    # See which members of kid get to keep their original symbols
    for k in kid:
        if kid[k] in codes: # if its code is valid and not already taken
            codes.remove(kid[k]) # remove it as an option
        else: # its code is not valid or has already been taken
            new_kid.pop(k) # make sure it gets a new code
    # Now, new_kid has symbols with their original codes, if valid

    # Next, assign remaining valid codes to remaining symbols
    for k in kid_syms:
        if k not in new_kid: # does not already have its symbol
            new_kid[k] = codes.pop()

    return new_kid
                

def crossover(mom, dad):
    """
    Function performing the crossover (and potential mutation) of "genes" from
    two parents to make two children
    Input: two parent codedicts (CodeDictionary)
    Output: two child codedicts (CodeDictionary)
    """
    logging.debug("Making children...")
    # Crossover point is approximately 1/2 of the symbols and codes
    cross = len(mom.symbols)//2

    # Randomly select mom and dad genes to pass down
    mom_genes = random.sample(mom.symbols,cross)
    dad_genes = random.sample(dad.symbols,cross)

    # start with genes from mom, complete with genes from dad
    kid1 = dict([(s[0],mom.codedict[s[0]]) for s in mom_ratios[:cross]])
    logging.debug("kid1 from mom:\n{}".format(kid1))
    try:
        kid1 = complete_dict(kid1.copy(), dad)
        kid1 = CodeDictionary(kid1)
        kid1.mutate() # check/execute mutation
        logging.debug("Kid 1:\n{}".format(kid1))
    except:
        logging.warning("complete_dict failed for kid1")
        # Getting a random code dictionary instead
        while True:
            try:
                kid1 = get_rand_codedict()
                break
            except:
                logging.warning("complete_dict failed")
                continue

    # start with genes from dad, complete with genes from mom
    kid2 = dict([(s[0],dad.codedict[s[0]]) for s in dad_ratios[:cross]])
    logging.debug("kid2 from dad:\n{}".format(kid2))
    try:
        kid2 = complete_dict(kid2.copy(), mom)
        kid2 = CodeDictionary(kid2)
        kid2.mutate() # check/execute mutation
        logging.debug("Kid 2:\n{}".format(kid2))
    except:
        logging.warning("complete_dict failed for kid2")
        # Getting a random code dictionary instead
        while True:
            try:
                kid2 = get_rand_codedict()
                break
            except Exception as e:
                logging.warning("complete_dict failed with Exception of " \
                                "type {}: {}".format(type(e),e))
                continue

    return kid1, kid2


def get_fitness(pop):
    """
    Function to calculate the fitness ratios of all codedicts in a population
    Input: a population (list) of codedicts (CodeDictionary)
    Output: a list of probabilities corresponding to the input population
    """
    logging.debug("Evaluating population...")
    xmax = max([x.comp_ratio for x in pop]) # ID the max codedict comp_ratio
    xmin = min([x.comp_ratio for x in pop]) # ID the min codedict comp_ratio
    if xmax==xmin: return [1/len(pop) for x in pop] # all codedicts are the same

    fitness = [((x.comp_ratio-xmin)/(xmax-xmin)) for x in pop] # initial scale
    fitness = [f/sum(fitness) for f in fitness] # adjust so sums up to 1
    logging.debug("Population fitnesses: {}".format(fitness))
    return fitness


def reproduce(pop):
    """
    Function to select parents according to their fitness within the population
    and get their children
    Input: a population (list) of codedicts (CodeDictionary)
    Output: the remaining population (list) minus the mom and dad,
            kid1 (CodeDictionary), and kid2 (CodeDictionary)
    """
    # If crossover hits, make new children
    if random.random() <= CodeDictionary.pc and len(pop) >= 2:
        logging.info("Reproducing children genetically...")
        popfit = get_fitness(pop)
        if popfit[:2] == [1.0, 0.0]: # Check for specific case
            mom, dad = pop.pop(0), pop.pop() # get the best choice and one other

        # Select 2, without replacement, with probabilities popfit
        else:
            mom, dad = nprandom.choice(len(popfit), size=2,
                                       replace=False, p=popfit)
            logging.debug("Mom, Dad indices: {}, {}".format(mom, dad))
            # Get the parents themselves, removing them from the population
            if mom > dad: mom, dad = pop.pop(mom), pop.pop(dad)
            else:         dad, mom = pop.pop(dad), pop.pop(mom)

        logging.debug("Mom:\n{}".format(mom))
        logging.debug("Dad:\n{}".format(dad))

        # Execute crossover
        kid1, kid2 = crossover(mom, dad)
        return pop, kid1, kid2

    # When crossover does not hit, return two randomly generated codedicts
    else:
        logging.info("Using random codedicts as children...")
        while True:
            try:
                kid1 = get_rand_codedict()
                break
            except Exception as e:
                logging.warning("complete_dict failed with Exception of " \
                                "type {}: {}".format(type(e),e))
                continue
        while True:
            try:
                kid2 = get_rand_codedict()
                break
            except Exception as e:
                logging.warning("complete_dict failed with Exception of " \
                                "type {}: {}".format(type(e),e))
                continue
        logging.debug("Kid 1:\n{}".format(kid1))
        logging.debug("Kid 2:\n{}".format(kid2))
        return pop, kid1, kid2

def get_next_pop(pop):
    """
    Function to take a population and from it produce the next population
    The DISCARD variable determines the portion of the population replaced in
    each generation
    Input: a population (list) of codedicts (CodeDictionary)
    Output: a population (list) of codedicts (CodeDictionary)
    """
    next_pop = pop.copy()

    # Get children through genetic reproduction
    while len(next_pop) < len(pop)+DISCARD: # make as many children as we need
        if len(pop) >= 2: # if there are valid parents, try and make children
            pop, kid1, kid2 = reproduce(pop)

            # Prevent duplicates from existing in the population
            for p in next_pop: 
                if kid1 == p: # Make use of class's custom function 
                    logging.info("kid1 is a duplicate; not keeping it")
                    break
            else: next_pop.append(kid1)

            for p in next_pop: 
                if kid2 == p: # Make use of class's custom function
                    logging.info("kid2 is a duplicate; not keeping it")
                    break
            else: next_pop.append(kid2)

        else: # try and increase genetic diversity if necessary
            logging.info("Adding random codedict to children")
            while True:
                try:
                    temp = get_rand_codedict()
                    break
                except Exception as e:
                    logging.warning("complete_dict failed with Exception of " \
                                    "type {}: {}".format(type(e),e))
                    continue
            next_pop.append(temp)

    logging.debug("Children: {}".format(next_pop[POP_SIZE:]))
    # Mix the children into the population, according to fitness
    next_pop = sorted(next_pop, key=attrgetter('comp_ratio'), reverse=True)
    # Keep only the most fit of the population
    next_pop = next_pop[:POP_SIZE]

    return next_pop


def performance(func):
    """
    Wrapper function to capture runtime and statistics
    Input: a function (whose performance will be measured)
    Output: a function (returning run time, each generation's achieved
                        compression ratios, and the best result found by the
                        algorithm)
    """
    def wrapper():
        start_time = time.time()
        gen_comp_ratios, result = func()
        end_time = time.time()
        return end_time - start_time, gen_comp_ratios, result
    return wrapper


@performance
def algorithm():
    """
    Function to make all the generations according to a Genetic Algorithm
    Input: None
    Output: list of each generation's compression ratios (list[float]) for use in
            performance statistics of the algorithm, the codedict (CodeDictionary)
            with the best compression ratio at the completion of the algorithm
    """
    gen_comp_ratios = []
    for i in range(NUM_GENS):
        logging.info("Getting population {}...".format(i))
        if i==0: mypop = get_init_pop()
        else:    mypop = get_next_pop(mypop)

        mypop = sorted(mypop, key=attrgetter('comp_ratio'), reverse=True)
        if mylevel == logging.DEBUG or i%5==0:
            print("Population:")
            for codedict in mypop: print(codedict)
        else:
            print("Best of Population {}:\n{}".format(i,mypop[0]))

        gen_comp_ratios.append([m.comp_ratio for m in mypop])
        logging.info("Population {} has {} members.".format(i,len(mypop)))

    result = mypop[0]
    return gen_comp_ratios, result


def pack(result):
    """
    Function to take a CodeDictionary and construct the binary output package
    Input: (CodeDictionary)
    Output: compressed product to be written to disk, sent via network, etc.
            (bytes)
    """
    # Header Part 1: nsymd_bits
    nsymd_bits = MAX_SYM_LEN.bit_length()
    header = bitarray('0')*(4-nsymd_bits.bit_length()) + \
             bitarray(format(nsymd_bits,'b'))
    if MAX_SYM_LEN > 15: raise Exception("Header nsymd_bits problem")
    logging.debug("Header with nsymd_bits {}".format(header))

    # Header Part 2: ncode_bits
    ncode_bits = 0 
    for s in result.codedict:
        ncode_bits = max(result.codedict[s].length(), ncode_bits)
    logging.debug("Maximum code bit size found: {}".format(ncode_bits))
    header += bitarray('0')*(4-ncode_bits.bit_length()) + \
              bitarray(format(ncode_bits,'b'))
    if ncode_bits > (2**4)-1: raise Exception("Header ncode_bits problem")
    logging.debug("Header with ncode_bits {}".format(header))

    # Header Part 3: nsyms
    nsyms = len(result.symbols)
    header += bitarray('0')*(10-nsyms.bit_length()) + \
              bitarray(format(nsyms,'b'))
    if nsyms > (2**10)-1: raise Exception("Header nsyms problem")
    logging.debug("Header with nsyms {}".format(header))

    # Header Part 4: npad_bits
    # ...defined and added after the body has been calculated
    logging.debug("Header npad_bits will be added later...")
        
    # Add the body: all nsymbs, symbols, ncodebs, and codes

    body = bitarray() # initialize
    for i in range(len(result.symbols)):
        
        # Body Part 1: nsymd
        nsymd = len(result.symbols[i]) # number of symbol digits
        body += bitarray('0')*(nsymd_bits-nsymd.bit_length()) + \
               bitarray(format(nsymd,'b'))

        # Body Part 2: the symbol
        for d in result.symbols[i]: # for the symbol
            body += bitarray('0')*(4-int(d).bit_length()) + \
                    bitarray(format(int(d),'b')) # each digit
        
        # Body Part 3: ncodeb
        ncodeb = result.codedict[result.symbols[i]].length() # number code bits
        body += bitarray('0')*(ncode_bits-ncodeb.bit_length()) + \
                bitarray(format(ncodeb,'b')) 

        # Body Part 4: the code
        body += result.codedict[result.symbols[i]] # the code
    logging.debug("Body code:\n{}".format(body))

    # Add Header Part 4
    npad_bits = 8-((21+body.length()+result.outdata.length())%8)
    header += bitarray('0')*(3-npad_bits.bit_length()) + \
              bitarray(format(npad_bits,'b')) # number of trailing zeros
    logging.debug("Header with npad_bits {}".format(header))

    logging.debug("Header length in bits: {}".format(header.length()))
    logging.debug("Body length in bits: {}".format(body.length()))
    logging.debug("Outdata length in bits: {}".format(result.outdata.length()))
    logging.debug("Number of padding bits: {}".format(npad_bits))

    # Complete
    outbitarray = header + body + result.outdata + bitarray('0')*npad_bits
    outbytes = outbitarray.tobytes()

    logging.debug("Total output: {}".format(outbytes))
    logging.debug("Total length in bytes: {}".format(len(outbytes)))

    return outbytes


def compare_compress(result):
    """
    Function to compare the algorithm's compression to other popular compression
    methods
    Input: The best result CodeDictionary from the algorithm
    Output: Relevant statistics
    """
    logging.debug("JSONing dictionary and outdata...")
    result_serial = dict([(s,result.codedict[s].to01())
                          for s in result.codedict])
    # dict{str:str} -> ujson-str -> utf-8 bytes
    result_jsonb = str.encode(ujson.dumps((result_serial,result.outdata.to01())))
    logging.debug("JSON-bytes length: {}".format(len(result_jsonb)))

    stats = [len(zlib.compress(result_jsonb)),
             len(bz2.compress(result_jsonb)),
             len(lzma.compress(result_jsonb)),
             len(zlib.compress(str.encode(INSTR))),
             len(bz2.compress(str.encode(INSTR))),
             len(lzma.compress(str.encode(INSTR))),
             len(zlib.compress(INBYTES)),
             len(bz2.compress(INBYTES)),
             len(lzma.compress(INBYTES))]

    return stats
    


################################# MAIN PROGRAM ##################################

print("Welcome to Data Compression with a Genetic Algorithm")

# Specify desired logging statements
mylevel = logging.INFO
logging.basicConfig(level=mylevel, format='%(levelname)s: %(message)s')
random.seed(0) 

# Define certain global variables
POP_SIZE = 20 # the number of members of the population
NUM_GENS = 20 # the number of generations each run should have
DISCARD = math.ceil(POP_SIZE/3) # replace ~1/3 of each generation's population
MAX_SYM_LEN = 8 # define maximum symbol length allowed

# Define the input integer
INDATA='12345678910'*400
INDATA_NUM_DIGITS=len("%i"%INDATA)
INSTR = str(INDATA) # need string version to use digits
INBYTES = INDATA.to_bytes(math.ceil(INDATA.bit_length()/8),byteorder='big')
logging.info("INDATA is {}".format(INDATA))

# Execute the algorithm, with @performance capturing desired statistics
runtime, gen_comp_ratios, result = algorithm()
print("\nBest Result:\n{}".format(result))

#################### COMPRESSION COMPARISON AND STATISTICS #####################

logging.debug("Packing results...")

print("\nAlgorithm runtime: {:.6f} seconds".format(runtime))
print("Compression ratio achieved: {:.8f}".format(result.comp_ratio))

logging.info("")
logging.info("Original data size           : {}".format(len(INBYTES)))
logging.info("Package compressed custom    : {}".format(len(result.outbytes)))

stats = compare_compress(result)
logging.info("")
logging.info("Package compressed with zlib : {}".format(stats[0]))
logging.info("Package compressed with bz2  : {}".format(stats[1]))
logging.info("Package compressed with lzma : {}".format(stats[2]))

logging.info("")
logging.info("Original str data compressed with zlib : {}".format(stats[3]))
logging.info("Original str data compressed with bz2  : {}".format(stats[4]))
logging.info("Original str data compressed with lzma : {}".format(stats[5]))
logging.info("")
logging.info("Original int data compressed with zlib : {}".format(stats[6]))
logging.info("Original int data compressed with bz2  : {}".format(stats[7]))
logging.info("Original int data compressed with lzma : {}".format(stats[8]))


# Process the generational information for statistics
mins, maxs, avgs = [], [], []
for i in gen_comp_ratios:
    mins.append(min(i))
    maxs.append(max(i))
    avgs.append(sum(i)/len(i))

pltname = 'plt_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.png'

with open('stats.csv','a',newline='') as f:
    statwriter = csv.writer(f)
    statwriter.writerow([INDATA_NUM_DIGITS, MAX_SYM_LEN, POP_SIZE,
                         NUM_GENS, runtime, result.comp_ratio,
                         len(INBYTES),len(result.outbytes),pltname] + stats + \
                        [mins] + [maxs] + [avgs] + [INDATA, result.codedict])

################################ PLOTTING GRAPHS ################################

logging.info("Plotting statistical data...")
plt.figure()
plt.xlabel('Generations')
plt.ylabel('Comp Ratios')
plt.title('GA Performance')
plt.plot(mins,'r') # Mininum, "worst" compression ratios plotted with red line
plt.plot(maxs,'g') # Maximum, "best" compression ratios plotted with green line
plt.plot(avgs,'b') # Average distances plotted with blue line
plt.savefig('Images/'+pltname)
