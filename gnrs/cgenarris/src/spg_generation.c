#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "read_input.h"
#include "spg_generation.h"
#include "combinatorics.h"
#include "lattice_generator.h"
#include "algebra.h"
#include "crystal_utils.h"
#include "molecule_utils.h"
#include "molecule_placement.h"
#include "spglib.h"

#define PI 3.141592653
extern unsigned int *seed2;

int generate_crystal(crystal* random_crystal, molecule* mol,float volume,
    float Z, float Zp_max, int spg, COMPATIBLE_SPG compatible_spg[],
    int len_compatible_spg, int compatible_spg_index,
    float norm_dev, float angle_std)
{
    Zp_max = 192; //stupid argument
    len_compatible_spg += 1; // not needed now; stupid argument

    random_crystal->Z = Z;
    int N = mol->num_of_atoms;

    //copy molecules to an array to save it. molecule might deform
    //upon many rotations.
    float Xm[N]; //molecule X coordinate
    float Ym[N];
    float Zm[N];
    copy_positions_to_array(mol, Xm, Ym, Zm);

    int hall_number;
    hall_number = hall_number_from_spg(spg);

    generate_lattice(random_crystal->lattice_vectors, spg, norm_dev, angle_std, volume);
    //find a random pos
    int pos_index = rand_r(seed2) % compatible_spg[compatible_spg_index].num_allowed_pos;
    int pos = compatible_spg[compatible_spg_index].allowed_pos[pos_index];
    random_crystal->wyckoff_position = pos;

    //place, align and attempt to generate crystal at position pos
    int result = auto_align_and_generate_at_position(random_crystal,
                            mol,
                            hall_number,
                            spg,
                            pos_index,
                            compatible_spg[compatible_spg_index]);
    //copy back to mol
    copy_positions_to_mol(mol, Xm, Ym, Zm);

    if(!result)
    {
        return 0;
    }
    else
        return 1;
}


// Target structure count per spg for the "pg_graded" distribution, indexed
// by spg number (1-230). Directly from CCDC "CSD Space Group Statistics -
// Space Group Frequency Ordering" (1 Jan 2025, 1,359,039 structures): the
// top 20 individual space groups by raw frequency (NOT point-group
// aggregated) are:
//   rank  spg  SG            rank  spg  SG
//    1    14   P21/c          11    5    C2
//    2    2    P-1            12    60   Pbcn
//    3    15   C2/c           13    148  R-3
//    4    19   P212121        14    29   Pca21
//    5    4    P21            15    13   P2/c
//    6    61   Pbca           16    12   C2/m
//    7    33   Pna21          17    7    Pc
//    8    9    Cc             18    11   P21/m
//    9    1    P1             19    18   P21212
//    10   62   Pnma           20    88   I41/a
// Target tapers quadratically (not linearly) from 500 (rank 1) down to 1
// (rank 20): target(rank) = round(500 * ((21-rank)/20)^2). The other 210
// space groups get 0.
static const int pg_graded_target[230] = {
    180, 451,   0, 320, 125,   0,  20,   0, 211,   0,
     11,  31,  45, 500, 405,   0,   0,   5, 361,   0,
      0,   0,   0,   0,   0,   0,   0,   0,  61,   0,
      0,   0, 245,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0, 101,
    281, 151,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,  80,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   1,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
};

int find_num_structure_for_spg(int num_structures, char spg_dist_type[10], int spg, int Z)
{
    if ( strcmp(spg_dist_type, "uniform") == 0)
        return num_structures;

    else if ( strcmp(spg_dist_type, "standard") == 0)
    {
        int order  = spg_positions[spg-1].multiplicity[0]/ Z;
        return (num_structures/order);
    }

    // list from Genarris 1.0 source code
    else if ( strcmp(spg_dist_type, "chiral") == 0 )
    {
        if (  spg == 1 || (spg > 2 && spg < 6) || (spg > 15 && spg < 25) ||
             (spg > 74 && spg < 81) || (spg > 88 && spg < 99) || (spg > 142 && spg < 147) ||
             (spg > 148 && spg < 156) || (spg > 167 && spg < 174) ||(spg > 176 && spg < 183) ||
             (spg > 194 && spg < 200) || (spg > 206 && spg < 215)
            )
        {
            return num_structures;
        }

        else
        {
            return 0;
        }
    }

    else if (strcmp(spg_dist_type, "racemic") == 0)
    {
        if (    spg == 2 || (spg > 5 && spg < 16) || (spg > 24 && spg < 75) ||
             (spg > 80 && spg < 89) || (spg > 98 && spg < 143) || (spg > 146 && spg < 149) ||
             (spg > 155 && spg < 168) ||(spg > 173 && spg < 177) || ( spg > 184 && spg < 195) ||
               (spg > 199 && spg < 207) || (spg > 214)
             )
            {
                return num_structures;
            }

            else
            {
                return 0;
            }

    }

    else if ( strcmp(spg_dist_type, "csd") == 0 )
    {
        const int len = 10;
        // list of high frequency spgs in csd
        // http://pd.chem.ucl.ac.uk/pdnn/symm3/sgpfreq.htm
        int csd_list[ ] = { 14, 2, 19, 15, 3, 61, 62, 33, 9, 1};
        for (int i = 0; i < len; i++)
        {
            if (csd_list[i] == spg)
                return num_structures;
        }

        return 0;
    }

    // "pg_graded" or "pg_graded:<scale>" (e.g. "pg_graded:0.1" tapers the
    // same CSD-frequency-shaped targets down to 1/10th, for fast
    // smoke-testing -- scale is passed in from Python, no recompile needed
    // to change it. Default scale is 1.0 (identical to plain "pg_graded").
    else if ( strncmp(spg_dist_type, "pg_graded", 9) == 0 )
    {
        int target = pg_graded_target[spg-1];
        if (target == 0)
            return 0;
        double scale = 1.0;
        if (spg_dist_type[9] == ':')
            scale = atof(spg_dist_type + 10);
        int scaled = (int)(target * scale + 0.5);
        if (scaled < 1)
            scaled = 1;
        return scaled;
    }

    else if(strcmp(spg_dist_type, "custom") == 0)
    {
        int custom[230], ret;
        int nspg = 0;
        FILE *fptr = fopen("spg", "r");

	if(!fptr)
	{
	    printf("spg file not found\n");
	    exit(EXIT_FAILURE);
	}
        do
        {
            ret = fscanf(fptr, "%d", &custom[nspg]);
            nspg++;

        }while(ret != EOF);
        nspg--;

        for(int j = 0; j < nspg; j++)
        {
            if(custom[j] == spg)
                return num_structures;
        }

        return 0;
    }

    else
    {
        printf("***ERROR: spg_generation: spg_dist_type not found\n");
        exit(EXIT_FAILURE);
    }
}
