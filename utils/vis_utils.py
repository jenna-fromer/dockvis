"""Interactive molecule visualization and filtering dashboard with RDKit and ipywidgets."""
import json 
import ipywidgets as widgets
import pandas as pd
from typing import Union
from io import BytesIO
from pathlib import Path
from tqdm import tqdm 
from PIL import Image, ImageDraw, ImageFont, ImageChops
from IPython.display import display, clear_output
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger
from multiprocessing import Pool, cpu_count


from .rxn_vis import visualize_reactions, reactions_from_syn
from . import prolif_utils

RDLogger.DisableLog('rdApp.*') 

def mol_from_smiles(smi):
    return Chem.MolFromSmiles(smi)

def clogp(mol): 
    return Crippen.MolLogP(mol)

def initialize_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Precompute molecular descriptors and auxiliary columns.

    Inputs:
        df (pd.DataFrame): Dataset with SMILES and related columns.

    Outputs:
        tuple[pd.DataFrame, pd.DataFrame]: (Original DataFrame, filtered copy).
    """
    # mols = [Chem.MolFromSmiles(smi) for smi in tqdm(df['SMILES'], desc='Chem.MolToSmiles')]
    n_proc = 16

    with Pool(n_proc) as pool:    
    
        mols = list(tqdm(
            pool.imap(mol_from_smiles, df['SMILES'], chunksize=5000), 
            total=len(df), desc='Chem.MolFromSmiles'
        ))

    guanidine = Chem.MolFromSmarts('[N]~[C](~[N])~[N]')
    df['Contains guanidium'] = [
        m.HasSubstructMatch(guanidine)
        for m in tqdm(mols, desc='Contains guanidinium')
    ]

    aminoimidazole = Chem.MolFromSmarts('[N]~[c](~[n])~[n]')
    df['Contains 2-aminoimidazole'] = [
        m.HasSubstructMatch(aminoimidazole)
        for m in tqdm(mols, desc='Contains 2-aminoimidazole')
    ]

    with Pool(n_proc) as pool:    

        df['cLogP'] = list(tqdm(
            pool.imap(clogp, mols, chunksize=5000), 
            total=len(df), desc='Crippen.cLogP'
        ))
        # [Crippen.MolLogP(m) for m in tqdm(mols, desc='cLogP')]
    
    names = {
        p.stem.split('_')[-1] for p in Path('data/redocked').glob('call_*')
    }
    df['Redocked'] = [
        f'data/redocked/call_{name}.sdf'
        if str(name) in names else None
        for name in df['Oracle call']
    ]

    with open('data/projection_buffer.json', 'r') as f:
        projection_buffer = json.load(f)
    
    syntheses = {val[0]: val[1] for val in projection_buffer.values()}
    df['Synthesis'] = [
        syntheses.get(smi, '')
        for smi in tqdm(df['SMILES'], desc='Getting synthesis')
    ]

    return df

def mol_to_image(
    smiles: str,
    size: tuple[int, int] = (200, 150),
    crop: bool = True,
    linewidth: int = 2,
    atom_labels: bool = True,
    coords_3d: bool = False,
    highlight_atom_dict: dict = None,
    mapped_smiles: str  = None,
    transparent: bool = True
) -> Union[Image.Image, None]:
    """
    Render a molecule SMILES to a cropped RDKit 2D/3D image.

    Inputs:
        smiles (str): Molecule SMILES.
        size (tuple[int, int]): Output image size (width, height).
        crop (bool): Crop whitespace automatically.
        linewidth (int): Line thickness for bonds.
        atom_labels (bool): Whether to display atom symbols.
        coords_3d (bool): Generate 3D coordinates for depiction.
        highlight_atom_dict (dict): Atom-color mapping.
        mapped_smiles (str): SMILES with atom mapping numbers.
        transparent (bool): Make background transparent.

    Outputs:
        Image.Image | None: Molecule image or None if parsing fails.
    """

    if highlight_atom_dict and mapped_smiles:
        mol = Chem.MolFromSmiles(mapped_smiles)
        fixed_highlight_dict = {}
        for atom in mol.GetAtoms():
            mapnum = atom.GetAtomMapNum()
            idx = atom.GetIdx()
            if mapnum in highlight_atom_dict:
                fixed_highlight_dict[idx] = highlight_atom_dict[mapnum]
            atom.SetAtomMapNum(0) 

        bond_highlight_dict = {}
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()

            if a1 in fixed_highlight_dict and a2 in fixed_highlight_dict:    
                if fixed_highlight_dict[a1] == fixed_highlight_dict[a2]:
                    bond_highlight_dict[bond.GetIdx()] = fixed_highlight_dict[a1] 

    else: 
        mol = Chem.MolFromSmiles(smiles)
        if coords_3d: 
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            mol = Chem.RemoveHs(mol)

    if mol is None:
        return None
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.baseFontSize = 1
    opts.bondLineWidth = linewidth
    opts.additionalAtomLabelPadding = 0.1
    opts.centreMoleculesBeforeDrawing = True
    opts.prepareMolsBeforeDrawing = True
    opts.fixedBondLength = 20
    opts.maxFontSize=18
    if not atom_labels: 
        opts.noAtomLabels = True 
    # better 2d visualization using 3d conformation 
    if coords_3d: 
        try: 
            AllChem.GenerateDepictionMatching3DStructure(mol, mol)
        except: 
            pass 
    
    if highlight_atom_dict and mapped_smiles:
        drawer.DrawMoleculeWithHighlights(mol,'', fixed_highlight_dict, bond_highlight_dict, {}, {})
    else: 
        drawer.DrawMolecule(mol)

    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()
    img = Image.open(BytesIO(png_bytes))

    # Convert to RGBA and make background transparent
    img = img.convert("RGBA")
    if transparent: 
        datas = img.getdata()
        newData = []
        for item in datas:
            # Replace white with transparent
            if item[0] > 240 and item[1] > 240 and item[2] > 240:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)
        img.putdata(newData)
    if crop: 
        bg = Image.new(img.mode, img.size, img.getpixel((0,0)))
        diff = ImageChops.difference(img, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        img = img.crop(bbox)
    return img

def mol_with_metadata(
        row: pd.Series,
        size: tuple[int, int] = (200, 150),
        font_size: int = 16,
        linewidth: float = 2,
        show_tautomer: bool = False
    ) -> Union[Image.Image, None]:
    """
    Render a molecule image with a compact metadata bar (objective, docking, interactions).

    Inputs:
        row (pd.Series | dict): Row with molecule and metadata columns.
        size (tuple[int, int]): Molecule image size.
        font_size (int): Font size for metadata labels.
        linewidth (float): Bond line width.
        show_tautomer (bool): If True, use 'Best conformer' SMILES.

    Outputs:
        Image.Image | None: Composite molecule + metadata image.
    """

    # --- 1. Draw molecule ---
    mol_img = mol_to_image(
        row['Best conformer'] if show_tautomer else row['SMILES'], 
        size=size, transparent=False, linewidth=linewidth)
    if mol_img is None:
        return None

    # --- 2. Extract relevant metadata ---
    obj = round(float(row["Objective"]), 1)
    dock = round(float(row["Docking score"]), 1)
    interacts_asp = int(row.get("Interacts with ASP", 0)) == 1
    interacts_water = int(row.get("Interacts with water", 0)) == 1

    # --- 3. Prepare info bar layout ---
    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    padding = 6
    box_height = font_size + 2 * padding

    # Prepare items to draw (text, fill color, font color, shape)
    items = [
        (f"{obj}", (230, 230, 230), (0, 0, 0), "rect"),  # light gray background, black text
        (f"{dock}", (255, 230, 230), (180, 0, 0), "rect")     # light red background, red text
    ]

    if interacts_asp:
        items.append(("ASP", (220, 245, 220), (0, 100, 0), "rect"))
    if interacts_water:
        items.append(("W", (220, 235, 255), (0, 70, 200), "rect"))
    if row['Redocked']: 
        items.append(("D", (255, 250, 210), (160, 120, 0), "rect"))

    # Measure total width
    draw_tmp = ImageDraw.Draw(mol_img)
    total_width = sum(draw_tmp.textbbox((0, 0), text, font=font)[2] - draw_tmp.textbbox((0, 0), text, font=font)[0] + 2 * padding for text, _, _, _ in items)
    total_width += (len(items) - 1) * 10  # spacing between boxes

    # Ensure total width fits molecule width (center-align later)
    bar_width = max(mol_img.width, total_width)
    bar_height = box_height + 2 * padding
    bar_img = Image.new("RGBA", (bar_width, bar_height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(bar_img)

    # --- 4. Draw items horizontally ---
    x = (bar_width - total_width) // 2
    for text, fill_color, font_color, shape in items:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        w = text_w + 2 * padding
        h = box_height

        if shape == "rect":
            draw.rounded_rectangle([x, (bar_height - h) / 2, x + w, (bar_height + h) / 2],
                                   radius=6, fill=fill_color)
        else:  # oval
            draw.ellipse([x, (bar_height - h) / 2, x + w, (bar_height + h) / 2],
                         fill=fill_color)

        # Center text
        text_x = x + (w - text_w) / 2
        text_y = (bar_height - text_h) / 2
        draw.text((text_x, text_y), text, fill=font_color, font=font)

        x += w + 10  # move right for next box

    # --- 5. Combine molecule and bar ---
    extra_padding = 10  # <--- add this for space below the labels
    total_height = mol_img.height + bar_img.height + extra_padding

    composite = Image.new("RGBA", (bar_width, total_height), (255, 255, 255, 255))
    composite.paste(mol_img, ((bar_width - mol_img.width) // 2, 0))
    composite.paste(bar_img, (0, mol_img.height))

    # --- 6. Optional: add top annotations ---
    draw = ImageDraw.Draw(composite)
    rank_text = f"Rank {row['Rank']}"
    oracle_text = f"#{row['Oracle call']:,}"

    padding_top = 5
    draw.text((padding_top, padding_top), rank_text, fill=(0, 0, 0), font=font)
    bbox = draw.textbbox((0, 0), oracle_text, font=font)
    x = composite.width - bbox[2] - padding_top
    draw.text((x, padding_top), oracle_text, fill=(0, 0, 0), font=font)

    return composite

def load_reaction_templates(txt_file: str = None): 
    """ Load a list of reaction templates from a text file. """
    txt_file = txt_file or 'data/rxn_templates/comprehensive.txt'
    with open('data/rxn_templates/comprehensive.txt', 'r') as f:
        lines = f.readlines()
        RXN_LIST = [line.strip() for line in lines]
    return RXN_LIST

def mol_with_metadata_wordy(
    row: pd.Series,
    size: tuple[int, int] = (200, 150),
    font_size: int = 16,
    linewidth: float = 2,
    show_tautomer: bool = False
) -> Union[Image.Image, None]:
    """Render a molecule image with a detailed metadata text panel.

    Inputs:
        row (pd.Series | dict): Molecule data row.
        size (tuple[int, int]): Molecule image size.
        font_size (int): Font size for metadata lines.
        linewidth (float): Bond line width.
        show_tautomer (bool): Use tautomer form if True.

    Outputs:
        Image.Image | None: Molecule image with detailed metadata.
    """
    # --- 1. Make molecule image ---
    mol_img = mol_to_image(
        row['Best conformer'] if show_tautomer else row['SMILES'], 
        size=size, transparent=False, linewidth=linewidth)
    if mol_img is None:
        return None

    # --- 2. Format metadata text ---
    obj = round(float(row["Objective"]), 1)
    dock = round(float(row["Docking score"]), 1)
    charge = int(row["Charge"])
    charge_str = f"{'+' if charge > 0 else ''}{charge}"
    
    check = 'Yes' # "[+]"
    cross = 'No' # "[-]"

    water_str = check if int(row["Interacts with water"]) == 1 else cross
    asp_str = check if int(row["Interacts with ASP"]) == 1 else cross

    # Compose lines (key : value pairs)
    metadata_lines = [
        f"Objective: {obj}",
        f"Docking score: {dock}",
        f"Charge: {charge_str}",
        f"Max rotatable bonds: {row['Max consecutive rotatable bonds']}",
        f"Unbound H donors: {row['Unbound H bond donors']}",
        f"Interacts with water: {water_str}",
        f"Interacts with ASP: {asp_str}",
    ]

    # --- 3. Compute layout ---
    width = mol_img.width
    line_height = font_size + 2
    text_height = line_height * len(metadata_lines) + 10
    total_height = mol_img.height + text_height

    # --- 4. Create composite canvas ---
    composite = Image.new("RGBA", (width, total_height), (255, 255, 255, 255))
    composite.paste(mol_img, (0, 0))

    # --- 5. Draw text panel ---
    draw = ImageDraw.Draw(composite)
    font = font = ImageFont.truetype("DejaVuSans.ttf", font_size) # ImageFont.load_default()

    y_offset = mol_img.height + 5
    for line in metadata_lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text(((width - text_w) / 2, y_offset), line, fill=(0, 0, 0), font=font)
        y_offset += line_height

    # Texts to overlay
    top_left_text = f"Rank {row['Rank']}"
    top_right_text = f"#{row['Oracle call']:,}"

    padding = 5  # distance from edges

    # --- Top-left ---
    draw.text((padding, padding), top_left_text, fill=(0, 0, 0), font=font)

    # --- Top-right ---
    bbox = draw.textbbox((0, 0), top_right_text, font=font)
    text_w = bbox[2] - bbox[0]
    x = composite.width - text_w - padding
    y = padding
    draw.text((x, y), top_right_text, fill=(0, 0, 0), font=font)

    return composite

class MoleculeGridSelectorWithFilters:
    """ Interactive ipywidgets-based molecule browser with filters, pagination, and reaction visualization. """
    def __init__(self, df: pd.DataFrame, n_rows: int = 2, n_cols: int = 8):
        """
        df: pandas DataFrame with columns:
            'SMILES', 'Objective', 'Oracle call', 'Docking score', 'Charge', 
            'Max consecutive rotatable bonds', 'Unbound H bond donors',
            'Interacts with water', 'Interacts with ASP'
        """
        self.df = df
        self.filtered_df = df.copy()
        self.page_size = n_rows*n_cols
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.page = 0

        # filters
        self.objective_slider = widgets.FloatRangeSlider(
            value=[min(df['Objective']), max(df['Objective'])],
            min=min(df['Objective']),
            max=max(df['Objective']),
            step=0.1,
            description="Objective:",
            continuous_update=False,
            readout_format=".1f",
            layout=widgets.Layout(width="350px")
        )      
        self.docking_slider = widgets.FloatRangeSlider(
            value=[min(df['Docking score']), 0],
            min=min(df['Docking score']),
            max=0,
            step=0.1,
            description='Docking score',
            continuous_update=False,
            readout_format=".1f",
            layout=widgets.Layout(width="400px"),
            style={'description_width': '150px'} 
        )   
        self.logp_slider = widgets.FloatRangeSlider(
            value=[min(self.df['cLogP']), max(self.df['cLogP'])], 
            min=min(self.df['cLogP']),
            max=max(self.df['cLogP']),
            step=0.1,
            description='Crippen cLogP',
            continuous_update=False,
            readout_format=".1f",
            layout=widgets.Layout(width="400px"),
            style={'description_width': '150px'}             
        )
        score_row = widgets.HBox([
            self.objective_slider, self.docking_slider, self.logp_slider,
        ], layout=widgets.Layout(align_items="center"))

        self.charge_slider = widgets.IntRangeSlider(
            value=[min(df['Charge']), max(df['Charge'])],
            min=min(df['Charge']),
            max=max(df['Charge']),
            description='Charge',
            continuous_update=False,
            layout=widgets.Layout(width="350px")
        )     

        self.rot_bonds_slider = widgets.IntRangeSlider(
            value=[min(df['Max consecutive rotatable bonds']), max(df['Max consecutive rotatable bonds'])],
            min=min(df['Max consecutive rotatable bonds']),
            max=max(df['Max consecutive rotatable bonds']),
            description='Max consecutive rotatable bonds',
            continuous_update=False,
            readout_format=".0f",
            layout=widgets.Layout(width="500px"),
            style={'description_width': '200px'} 
        )    
        self.unbound_donors_slider = widgets.IntRangeSlider(
            value=[min(df['Unbound H bond donors']), max(df['Unbound H bond donors'])],
            min=min(df['Unbound H bond donors']),
            max=max(df['Unbound H bond donors']),
            step=1,
            description='Unbound H bond donors',
            continuous_update=False,
            readout_format=".0f",
            layout=widgets.Layout(width="500px"),
            style={'description_width': '200px'} 
        ) 
        more_sliders_row = widgets.HBox([
            self.charge_slider, self.rot_bonds_slider, self.unbound_donors_slider
        ], layout=widgets.Layout(align_items="center"))

        self.water_interaction_checkbox = widgets.Checkbox(
            value=False,
            indent=False,
            layout=widgets.Layout(width='45px', justify_content="flex-end")
        )
        self.water_interaction_box = widgets.HBox(
            [self.water_interaction_checkbox, widgets.Label("Interacts with water", width='500px', margin='0px 0 0 0px')], 
            layout=widgets.Layout(display="flex", flex_flow="row wrap", justify_content="flex-start")
        )

        self.asp_interaction_checkbox = widgets.Checkbox(
            value=False,
            indent=False,
            layout=widgets.Layout(width='45px', justify_content="flex-end")
        )
        self.asp_interaction_box = widgets.HBox(
            [self.asp_interaction_checkbox, widgets.Label("Interacts with asp residue", width='500px', margin='0px 0 0 0px')], 
            layout=widgets.Layout(display="flex", flex_flow="row wrap", justify_content="flex-start")
        )
        checkbox_row = widgets.HBox([
            widgets.Label("Interactions: "), 
            self.asp_interaction_box, 
            self.water_interaction_box,
        ], layout=widgets.Layout(align_items="center"))

        # Other filters 
        self.no_guan_checkbox = widgets.Checkbox(
            value=False,
            indent=False,
            layout=widgets.Layout(width='45px', justify_content="flex-end")
        )
        self.no_guan_box = widgets.HBox(
            [self.no_guan_checkbox, widgets.Label("No guanidinium", width='500px', margin='0px 0 0 0px')], 
            layout=widgets.Layout(display="flex", flex_flow="row wrap", justify_content="flex-start")
        )
        self.no_aminoimidazole_checkbox = widgets.Checkbox(
            value=False,
            indent=False,
            layout=widgets.Layout(width='45px', justify_content="flex-end")
        )
        self.no_aminoimidazole_box = widgets.HBox(
            [self.no_aminoimidazole_checkbox, widgets.Label("No 2-aminoimidazole", width='500px', margin='0px 0 0 0px')], 
            layout=widgets.Layout(display="flex", flex_flow="row wrap", justify_content="flex-start")
        )
        self.other_filters_row = widgets.HBox([
            widgets.Label("Other filters: "), 
            self.no_guan_box, 
            self.no_aminoimidazole_box,
        ], layout=widgets.Layout(align_items="center"))

        # --- Sorting choice ---
        self.sort_choice = widgets.ToggleButtons(
            options=[("Objective score", "Objective"), ("Docking score", "Docking score")],
            value="Objective",  # default
            description="Sort by:",
            button_style="",  # optional: can use 'primary'
            layout=widgets.Layout(width="400px"),
            style={'description_width': '80px'}
        )

        # --- Apply filters button ---
        self.apply_button = widgets.Button(
            description="Apply filters and sorting",
            button_style="primary",
            layout=widgets.Layout(width="200px", height="30px", margin="15px 0 0 0")
        )
        self.apply_button.on_click(lambda b: self.apply_filters())
        self.filter_output = widgets.Output(layout=widgets.Layout(min_height="25px"))

        # Filter UI
        self.filter_box = widgets.VBox([score_row, more_sliders_row, checkbox_row, self.other_filters_row, self.sort_choice, self.apply_button, self.filter_output])
        
        # Pagination controls
        self.output = widgets.Output()
        self.page_label = widgets.Label()
        self.first_button = widgets.Button(description="First", layout=widgets.Layout(width='80px'))
        self.prev_button = widgets.Button(description="Previous", layout=widgets.Layout(width='80px'))
        self.next_button = widgets.Button(description="Next", layout=widgets.Layout(width='80px'))
        self.last_button = widgets.Button(description="Last", layout=widgets.Layout(width='80px'))

        self.first_button.on_click(self.first_page)
        self.prev_button.on_click(self.prev_page)
        self.next_button.on_click(self.next_page)
        self.last_button.on_click(self.last_page)


        self.grid_box = widgets.GridBox(
            layout=widgets.Layout(
                display='grid',
                grid_template_columns=f'repeat({self.n_cols}, minmax(150px, 1fr))',
                grid_gap='10px',
                justify_content='flex-start',
                width='100%'
            )
        )

        self.legend_box = self.define_legend_widget()
        self.controls = widgets.HBox(
            [
                widgets.HBox(
                    [self.first_button, self.prev_button, self.page_label, self.next_button, self.last_button],
                    layout=widgets.Layout(justify_content="center", align_items="center")
                ),
                widgets.HBox(
                    [self.legend_box],
                    layout=widgets.Layout(justify_content="center", align_items="center")
                ),
            ],
            layout=widgets.Layout(justify_content="space-between", align_items="center", width="100%")
        )

        # Details display area
        self.details_box = widgets.Output(
            layout=widgets.Layout(
                padding='10px',
                min_height='120px',
                width='100%',
                margin='10px 0 0 0'
            )
        )

        self.container = widgets.VBox(
            [self.filter_box, self.controls, self.grid_box, self.details_box, self.output], 
            layout=widgets.Layout(width='100%')
        )

        with self.filter_output:
            clear_output(wait=True)
            print(f"{len(self.df):,} compounds before applying filters")
                  
        # Initial display
        self.update_grid()
    
    def initialize_df(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Precompute molecular descriptors and auxiliary columns.

        Inputs:
            df (pd.DataFrame): Dataset with SMILES and related columns.

        Outputs:
            tuple[pd.DataFrame, pd.DataFrame]: (Original DataFrame, filtered copy).
        """
        mols = [Chem.MolFromSmiles(smi) for smi in df['SMILES']]

        guanidine = Chem.MolFromSmarts('[N]~[C](~[N])~[N]')
        df['Contains guanidium'] = [
            m.HasSubstructMatch(guanidine)
            for m in mols
        ]

        aminoimidazole = Chem.MolFromSmarts('[N]~[c](~[n])~[n]')
        df['Contains 2-aminoimidazole'] = [
            m.HasSubstructMatch(aminoimidazole)
            for m in mols
        ]

        df['cLogP'] = [Crippen.MolLogP(m) for m in mols]
        
        df['Redocked'] = [
            f'data/redocked/call_{name}.sdf'
            if Path(f'data/redocked/call_{name}.sdf').exists() else None
            for name in df['Oracle call']
        ]

        with open('data/projection_buffer.json', 'r') as f:
            projection_buffer = json.load(f)
        
        syntheses = {val[0]: val[1] for val in projection_buffer.values()}
        df['Synthesis'] = [
            syntheses.get(smi, '')
            for smi in df['SMILES']
        ]

        filtered_df = df.copy()
        return df, filtered_df

    def define_legend_widget(self, size=(500, 25), font_size=10): 
        """ Generate a legend image widget explaining the metadata colors. """
        items = [
            ("Objective score", (230, 230, 230), (0, 0, 0), "rect"),
            ("Docking score", (255, 230, 230), (180, 0, 0), "rect"),
            ("Interacts with aspartate residue", (220, 245, 220), (0, 100, 0), "rect"),
            ("Interacts with water", (220, 235, 255), (0, 70, 200), "rect"),
            ("Pose available", (255, 250, 210), (160, 120, 0), "rect")
        ]

        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        draw_dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1), (255, 255, 255, 0)))
        padding = 6

        # --- Compute box dimensions dynamically ---
        text_widths = []
        for text, _, _, _ in items:
            bbox = draw_dummy.textbbox((0, 0), text, font=font)
            text_widths.append(bbox[2] - bbox[0])
        
        total_width = sum(w + 2 * padding for w in text_widths) + padding * (len(items) - 1)
        total_height = font_size + 4*padding

        # --- Create image and draw ---
        img = Image.new("RGBA", (total_width, total_height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        x = 0
        for i, (text, fill_color, font_color, shape) in enumerate(items):
            text_w = text_widths[i]
            w = text_w + 2 * padding
            h = font_size + 2 * padding

            if shape == "rect":
                draw.rounded_rectangle(
                    [x, (total_height - h) / 2, x + w, (total_height + h) / 2],
                    radius=6,
                    fill=fill_color
                )
            else:
                draw.ellipse(
                    [x, (total_height - h) / 2, x + w, (total_height + h) / 2],
                    fill=fill_color
                )

            text_x = x + (w - text_w) / 2
            text_y = padding*2
            draw.text((text_x, text_y), text, fill=font_color, font=font)

            x += w + padding

        # Convert to PNG bytes
        bio = BytesIO()
        img.save(bio, format="PNG")
        return widgets.Image(value=bio.getvalue())

    def apply_filters(self): 
        """ Filter dataset based on widget criteria and update the displayed molecule grid. """
        mask = (
            (self.df['Objective'] >= self.objective_slider.value[0]) & 
            (self.df['Objective'] <= self.objective_slider.value[1]) &
            (self.df['cLogP'] >= self.logp_slider.value[0]) & 
            (self.df['cLogP'] <= self.logp_slider.value[1]) & 
            (self.df['Docking score'] >= self.docking_slider.value[0]) &
            (self.df['Docking score'] <= self.docking_slider.value[1]) & 
            (self.df['Charge'] >= self.charge_slider.value[0]) &
            (self.df['Charge'] <= self.charge_slider.value[1]) & 
            (self.df['Max consecutive rotatable bonds'] >= self.rot_bonds_slider.value[0]) &
            (self.df['Max consecutive rotatable bonds'] <= self.rot_bonds_slider.value[1]) &
            (self.df['Unbound H bond donors'] >= self.unbound_donors_slider.value[0]) & 
            (self.df['Unbound H bond donors'] <= self.unbound_donors_slider.value[1])
        )

        self.filtered_df = self.df[mask]
        self.filtered_df.sort_values(by=self.sort_choice.value, inplace=True, ascending=False if self.sort_choice.value=='Objective' else True)
        self.filtered_df['Rank'] = range(1, len(self.filtered_df) + 1)

        if self.water_interaction_checkbox.value: 
            self.filtered_df = self.filtered_df.loc[self.filtered_df['Interacts with water'] == 1]
        
        if self.asp_interaction_checkbox.value: 
            self.filtered_df = self.filtered_df.loc[self.filtered_df['Interacts with ASP'] == 1]
        
        if self.no_guan_checkbox.value: 
            self.filtered_df = self.filtered_df.loc[self.filtered_df['Contains guanidium'] == False]

        if self.no_aminoimidazole_checkbox.value: 
            self.filtered_df = self.filtered_df.loc[self.filtered_df['Contains 2-aminoimidazole'] == False]

        # reset page 
        self.page = 0

        # update grid 
        max_page = max((len(self.filtered_df) - 1) // self.page_size, 0)
        start = 0
        end = min(start + self.page_size, len(self.filtered_df))
        self.grid_box.children = [self.mol_image_widget(row) for _, row in self.filtered_df.iloc[start:end].iterrows()]
        self.page_label.value = f"Page {self.page + 1} of {max_page + 1}"

        # Update summary output
        with self.filter_output:
            clear_output(wait=True)
            print(f"{len(self.filtered_df):,} compounds pass all filters")

    def mol_image_widget(self, row: pd.Series) -> widgets.Box:
        """
        Return a widget containing a molecule image and a 'Select' button.

        Inputs:
            row (pd.Series | dict): Row of molecule data.

        Outputs:
            widgets.Box: Molecule display box widget.
        """
        # --- 1. Generate molecule image ---
        img = mol_with_metadata(row, size=(350,250), linewidth=2, font_size=20, show_tautomer=True)
        if img is None:
            return widgets.Label(value="Invalid")

        # --- 2. Convert to Image widget ---
        bio = BytesIO()
        img.save(bio, format="PNG")
        image_widget = widgets.Image(
            value=bio.getvalue(),
            format='png',
            width=img.width,
            height=img.height,
            layout=widgets.Layout(padding='0px', margin='0px')
        )

        # --- 3. Create Select button ---
        select_button = widgets.Button(
            description="Select",
            layout=widgets.Layout(width='80px', height='25px')
        )
        select_button.on_click(lambda b, r=row: self.select_row(r))

        # --- 4. Combine image and button into a VBox ---
        vbox = widgets.VBox([image_widget, select_button], layout=widgets.Layout(align_items='center', padding='0px'))
        
        # --- 5. Add border around the entire VBox ---
        bordered_box = widgets.Box([vbox], layout=widgets.Layout(
            border='1px solid black',
            display='flex',
            flex_flow='column',
            align_items='center',
            padding='2px',
            margin='2px', 
        ))

        return bordered_box

    def update_grid(self):
        """ Update molecule grid display according to pagination and filters. """
        start = self.page * self.page_size
        end = min(start + self.page_size, len(self.filtered_df))
        rows = [self.filtered_df.iloc[i] for i in range(start, end)]
        self.grid_box.children = [self.mol_image_widget(row) for row in rows]

        self.page_label.value = f"Page {self.page + 1} of {((len(self.filtered_df)-1)//self.page_size)+1}"
        self.output.clear_output()

    def get_prolif(self, row: pd.Series):
        """ Generate 2D/3D interaction visualizations for a molecule. """
        if row['Redocked'] is None:
            return widgets.Label('No pose'), None
        view_3d, map_2d = prolif_utils.make_prolif_viewer(
            sdf_path=row['Redocked']
        )
        if map_2d is not None: 
            return widgets.HTML(map_2d.data), view_3d
        return widgets.Label('Failure to visualize pose'), None
         
    def select_row(self, row: pd.Series):
        """ Display detailed metadata, synthesis route, and interaction visualization for a selected molecule. """
        
        self.details_box.clear_output(wait=True)
        with self.details_box:
            # --- 1. Prepare metadata ---
            obj = row['Objective']
            dock = row['Docking score']
            water_str = 'Yes' if row['Interacts with water'] else 'No'
            asp_str = 'Yes' if row['Interacts with ASP'] else 'No'

            metadata_lines = [
                f"Objective: {obj:0.1f}",
                f"Docking score: {dock:0.1f}",
                f"Crippen cLogP: {row['cLogP']:0.1f}",
                f"Charge: {row['Charge']:+}",
                f"Max rotatable bonds: {row['Max consecutive rotatable bonds']}",
                f"Unbound H donors: {row['Unbound H bond donors']}",
                f"Interacts with water: {water_str}",
                f"Interacts with ASP: {asp_str}",
            ]

            # --- Wrap metadata lines in HTML for text wrapping ---
            metadata_widgets = [
                widgets.HTML(value=f"<p style='margin:0; padding:0; font-size:16px;'>{line}</span>")
                for line in metadata_lines
            ]
            metadata_box_content = widgets.VBox(
                metadata_widgets,
                layout=widgets.Layout(
                    display='flex',
                    flex_flow='column',
                    align_items='flex-start',
                    gap='0px',   # crucial: remove vertical space between children
                    padding='10px',
                    margin='0px'
                )
            )
            # --- Bordered metadata box with title ---
            metadata_box = widgets.VBox([
                widgets.HTML("<h2 style='text-align:center;margin:0'>Details</h2>"),
                metadata_box_content
            ], layout=widgets.Layout(
                border='1px solid black',
                width='18%',
                padding='10px',
                height='450px',
                overflow='hidden', 
            ))

            # --- 2. Generate synthesis reactions ---
            synthesis = row['Synthesis']  # assumes you have a 'Synthesis' column with the string

            if synthesis == 'StartingPool':
                rxn_widget = widgets.Label("In starting pool")
            else:
                try:
                    reactions = reactions_from_syn(synthesis)
                    if not reactions:
                        raise TypeError
                    rxn_img = visualize_reactions(reactions)
                except:
                    reactants = [r for r in synthesis.split(';') if not r.startswith('R')]
                    reaction = f"{'.'.join(reactants)}>>{row['SMILES']}"
                    rxn_img = visualize_reactions(
                        [reaction],
                        headers=['Visualization failed from string, showing as single step']
                    )

                bio = BytesIO()
                rxn_img.save(bio, format="PNG")
                rxn_widget = widgets.Image(
                    value=bio.getvalue(),
                    format='png',
                    layout=widgets.Layout(width='100%', height='auto', padding='0px', margin='0px')
                )

            # --- Synthesis box with centered header ---
            rxn_box = widgets.VBox([
                widgets.HTML("<h2 style='text-align:center;margin:0'>Synthesis</h2>"),
                rxn_widget
            ], layout=widgets.Layout(
                width='45%',
                align_items='center',
                border='1px solid black',
                padding='10px', height='450px'
            ))

            # --- 3. Combine metadata + spacer + reaction images ---
            widget_2d, view_3d = self.get_prolif(row)
            interaction_box = widgets.VBox([
                widgets.HTML("<h2 style='text-align:center;margin:0'>Interactions</h2>"),
                widget_2d
            ], layout=widgets.Layout(
                width='35%',
                align_items='center',
                border='1px solid black',
                padding='10px',
                height='450px'
            ))
            row_box = widgets.HBox(
                [
                    metadata_box,  # Left column
                    interaction_box, # Middle
                    rxn_box        # Right column
                ],
                layout=widgets.Layout(
                    width='100%',
                    align_items='flex-start',
                    justify_content='space-between',  
                    padding='0 0 100px 0'   # top, right, bottom, left padding
                )
            )

            
            display(row_box)

            if view_3d: 
                view_3d.show()

    # --- Pagination methods ---
    def prev_page(self, *args):
        if self.page > 0:
            self.page -= 1
            self.update_grid()

    def next_page(self, *args):
        if (self.page + 1) * self.page_size < len(self.filtered_df):
            self.page += 1
            self.update_grid()

    def first_page(self, *args):
        self.page = 0
        self.update_grid()

    def last_page(self, *args):
        self.page = (len(self.filtered_df)-1) // self.page_size
        self.update_grid()

    # --- Display the widget ---
    def display(self):
        display(self.container)

