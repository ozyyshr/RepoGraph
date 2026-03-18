import copy
import operator
import warnings
import weakref
from contextlib import suppress
from functools import reduce
import numpy as np
from numpy import char as chararray
from .column import ASCIITNULL, FITS2NUMPY, ASCII2NUMPY, ASCII2STR, ColDefs, _AsciiColDefs, _FormatX, _FormatP, _VLF, _get_index, _wrapx, _unwrapx, _makep, Delayed
from .util import decode_ascii, encode_ascii, _rstrip_inplace
from ...utils import lazyproperty
class FITS_record:
    def __init__(self, input, row=0, start=None, end=None, step=None, base=None, **kwargs):
        self.array = input
        self.row = row
        if base:
            width = len(base)
        else:
            width = self.array._nfields
        s = slice(start, end, step).indices(width)
        (self.start, self.end, self.step) = s
        self.base = base
    def __getitem__(self, key):
        if isinstance(key, str):
            indx = _get_index(self.array.names, key)
            if indx < self.start or indx > self.end - 1:
                raise KeyError("Key '{}' does not exist.".format(key))
        elif isinstance(key, slice):
            return type(self)(self.array, self.row, key.start, key.stop, key.step, self)
        else:
            indx = self._get_index(key)
            if indx > self.array._nfields - 1:
                raise IndexError('Index out of bounds')
        return self.array.field(indx)[self.row]
    def __setitem__(self, key, value):
        if isinstance(key, str):
            indx = _get_index(self.array.names, key)
            if indx < self.start or indx > self.end - 1:
                raise KeyError("Key '{}' does not exist.".format(key))
        elif isinstance(key, slice):
            for indx in range(slice.start, slice.stop, slice.step):
                indx = self._get_indx(indx)
                self.array.field(indx)[self.row] = value
        else:
            indx = self._get_index(key)
            if indx > self.array._nfields - 1:
                raise IndexError('Index out of bounds')
        self.array.field(indx)[self.row] = value
    def __len__(self):
        return len(range(self.start, self.end, self.step))
    def __repr__(self):
        outlist = []
        for idx in range(len(self)):
            outlist.append(repr(self[idx]))
        return '({})'.format(', '.join(outlist))
    def field(self, field):
        return self.__getitem__(field)
    def setfield(self, field, value):
        self.__setitem__(field, value)
    @lazyproperty
    def _bases(self):
        bases = [weakref.proxy(self)]
        base = self.base
        while base:
            bases.append(base)
            base = base.base
        return bases
    def _get_index(self, indx):
        indices = np.ogrid[:self.array._nfields]
        for base in reversed(self._bases):
            if base.step < 1:
                s = slice(base.start, None, base.step)
            else:
                s = slice(base.start, base.end, base.step)
            indices = indices[s]
        return indices[indx]
class FITS_rec(np.recarray):
    _record_type = FITS_record
    _character_as_bytes = False
    def __new__(subtype, input):
        if input.dtype.subdtype is None:
            self = np.recarray.__new__(subtype, input.shape, input.dtype, buf=input.data)
        else:
            self = np.recarray.__new__(subtype, input.shape, input.dtype, buf=input.data, strides=input.strides)
        self._init()
        if self.dtype.fields:
            self._nfields = len(self.dtype.fields)
        return self
    def __setstate__(self, state):
        meta = state[-1]
        column_state = state[-2]
        state = state[:-2]
        super().__setstate__(state)
        self._col_weakrefs = weakref.WeakSet()
        for (attr, value) in zip(meta, column_state):
            setattr(self, attr, value)
    def __reduce__(self):
        (reconst_func, reconst_func_args, state) = super().__reduce__()
        column_state = []
        meta = []
        for attrs in ['_converted', '_heapoffset', '_heapsize', '_nfields', '_gap', '_uint', 'parnames', '_coldefs']:
            with suppress(AttributeError):
                if attrs == '_coldefs':
                    column_state.append(self._coldefs.__deepcopy__(None))
                else:
                    column_state.append(getattr(self, attrs))
                meta.append(attrs)
        state = state + (column_state, meta)
        return (reconst_func, reconst_func_args, state)
    def __array_finalize__(self, obj):
        if obj is None:
            return
        if isinstance(obj, FITS_rec):
            self._character_as_bytes = obj._character_as_bytes
        if isinstance(obj, FITS_rec) and obj.dtype == self.dtype:
            self._converted = obj._converted
            self._heapoffset = obj._heapoffset
            self._heapsize = obj._heapsize
            self._col_weakrefs = obj._col_weakrefs
            self._coldefs = obj._coldefs
            self._nfields = obj._nfields
            self._gap = obj._gap
            self._uint = obj._uint
        elif self.dtype.fields is not None:
            self._nfields = len(self.dtype.fields)
            self._converted = {}
            self._heapoffset = getattr(obj, '_heapoffset', 0)
            self._heapsize = getattr(obj, '_heapsize', 0)
            self._gap = getattr(obj, '_gap', 0)
            self._uint = getattr(obj, '_uint', False)
            self._col_weakrefs = weakref.WeakSet()
            self._coldefs = ColDefs(self)
            for col in self._coldefs:
                del col.array
                col._parent_fits_rec = weakref.ref(self)
        else:
            self._init()
    def _init(self):
        self._nfields = 0
        self._converted = {}
        self._heapoffset = 0
        self._heapsize = 0
        self._col_weakrefs = weakref.WeakSet()
        self._coldefs = None
        self._gap = 0
        self._uint = False
    @classmethod
    def from_columns(cls, columns, nrows=0, fill=False, character_as_bytes=False):
        if not isinstance(columns, ColDefs):
            columns = ColDefs(columns)
        for column in columns:
            arr = column.array
            if isinstance(arr, Delayed):
                if arr.hdu.data is None:
                    column.array = None
                else:
                    column.array = _get_recarray_field(arr.hdu.data, arr.field)
        del columns._arrays
        if nrows == 0:
            for arr in columns._arrays:
                if arr is not None:
                    dim = arr.shape[0]
                else:
                    dim = 0
                if dim > nrows:
                    nrows = dim
        raw_data = np.empty(columns.dtype.itemsize * nrows, dtype=np.uint8)
        raw_data.fill(ord(columns._padding_byte))
        data = np.recarray(nrows, dtype=columns.dtype, buf=raw_data).view(cls)
        data._character_as_bytes = character_as_bytes
        columns._add_listener(data)
        data._coldefs = columns
        if fill:
            return data
        for (idx, column) in enumerate(columns):
            arr = column.array
            if arr is None:
                array_size = 0
            else:
                array_size = len(arr)
            n = min(array_size, nrows)
            if not n:
                continue
            field = _get_recarray_field(data, idx)
            name = column.name
            fitsformat = column.format
            recformat = fitsformat.recformat
            outarr = field[:n]
            inarr = arr[:n]
            if isinstance(recformat, _FormatX):
                if inarr.shape[-1] == recformat.repeat:
                    _wrapx(inarr, outarr, recformat.repeat)
                    continue
            elif isinstance(recformat, _FormatP):
                data._cache_field(name, _makep(inarr, field, recformat, nrows=nrows))
                continue
            elif recformat[-2:] == FITS2NUMPY['L'] and inarr.dtype == bool:
                field[:] = ord('F')
                converted = np.zeros(field.shape, dtype=bool)
                converted[:n] = inarr
                data._cache_field(name, converted)
                inarr = np.where(inarr == np.False_, ord('F'), ord('T'))
            elif columns[idx]._physical_values and columns[idx]._pseudo_unsigned_ints:
                bzero = column.bzero
                converted = np.zeros(field.shape, dtype=inarr.dtype)
                converted[:n] = inarr
                data._cache_field(name, converted)
                if n < nrows:
                    field[n:] = -bzero
                inarr = inarr - bzero
            elif isinstance(columns, _AsciiColDefs):
                if fitsformat._pseudo_logical:
                    outarr = field.view(np.uint8, np.ndarray)[:n]
                elif arr.dtype.kind not in ('S', 'U'):
                    data._cache_field(name, np.zeros(nrows, dtype=arr.dtype))
                    outarr = data._converted[name][:n]
                outarr[:] = inarr
                continue
            if inarr.shape != outarr.shape:
                if inarr.dtype.kind == outarr.dtype.kind and inarr.dtype.kind in ('U', 'S') and (inarr.dtype != outarr.dtype):
                    inarr_rowsize = inarr[0].size
                    inarr = inarr.flatten().view(outarr.dtype)
                if outarr.ndim > 1:
                    inarr_rowsize = inarr[0].size
                    inarr = inarr.reshape(n, inarr_rowsize)
                    outarr[:, :inarr_rowsize] = inarr
                else:
                    outarr[:n] = inarr.ravel()
            else:
                outarr[:] = inarr
        for idx in range(len(columns)):
            columns._arrays[idx] = data.field(idx)
        return data
    def __repr__(self):
        return np.ndarray.__repr__(self)
    def __getitem__(self, key):
        if self._coldefs is None:
            return super().__getitem__(key)
        if isinstance(key, str):
            return self.field(key)
        out = self.view(np.recarray)[key]
        if type(out) is not np.recarray:
            return self._record_type(self, key)
        out = out.view(type(self))
        out._coldefs = ColDefs(self._coldefs)
        arrays = []
        out._converted = {}
        for (idx, name) in enumerate(self._coldefs.names):
            arrays.append(self._coldefs._arrays[idx][key])
            if name in self._converted:
                dummy = self._converted[name]
                field = np.ndarray.__getitem__(dummy, key)
                out._converted[name] = field
        out._coldefs._arrays = arrays
        return out
    def __setitem__(self, key, value):
        if self._coldefs is None:
            return super().__setitem__(key, value)
        if isinstance(key, str):
            self[key][:] = value
            return
        if isinstance(key, slice):
            end = min(len(self), key.stop or len(self))
            end = max(0, end)
            start = max(0, key.start or 0)
            end = min(end, start + len(value))
            for idx in range(start, end):
                self.__setitem__(idx, value[idx - start])
            return
        if isinstance(value, FITS_record):
            for idx in range(self._nfields):
                self.field(self.names[idx])[key] = value.field(self.names[idx])
        elif isinstance(value, (tuple, list, np.void)):
            if self._nfields == len(value):
                for idx in range(self._nfields):
                    self.field(idx)[key] = value[idx]
            else:
                raise ValueError('Input tuple or list required to have {} elements.'.format(self._nfields))
        else:
            raise TypeError('Assignment requires a FITS_record, tuple, or list as input.')
    def copy(self, order='C'):
        new = super().copy(order=order)
        new.__dict__ = copy.deepcopy(self.__dict__)
        return new
    @property
    def columns(self):
        return self._coldefs
    @property
    def _coldefs(self):
        return self.__dict__.get('_coldefs')
    @_coldefs.setter
    def _coldefs(self, cols):
        self.__dict__['_coldefs'] = cols
        if isinstance(cols, ColDefs):
            for col in cols.columns:
                self._col_weakrefs.add(col)
    @_coldefs.deleter
    def _coldefs(self):
        try:
            del self.__dict__['_coldefs']
        except KeyError as exc:
            raise AttributeError(exc.args[0])
    def __del__(self):
        try:
            del self._coldefs
            if self.dtype.fields is not None:
                for col in self._col_weakrefs:
                    if col.array is not None:
                        col.array = col.array.copy()
        except (AttributeError, TypeError):
            pass
    @property
    def names(self):
        if self.dtype.fields:
            return list(self.dtype.names)
        elif getattr(self, '_coldefs', None) is not None:
            return self._coldefs.names
        else:
            return None
    @property
    def formats(self):
        if getattr(self, '_coldefs', None) is not None:
            return self._coldefs.formats
        return None
    @property
    def _raw_itemsize(self):
        if _has_unicode_fields(self):
            total_itemsize = 0
            for field in self.dtype.fields.values():
                itemsize = field[0].itemsize
                if field[0].kind == 'U':
                    itemsize = itemsize // 4
                total_itemsize += itemsize
            return total_itemsize
        else:
            return self.itemsize
    def field(self, key):
        column = self.columns[key]
        name = column.name
        format = column.format
        if format.dtype.itemsize == 0:
            warnings.warn('Field {!r} has a repeat count of 0 in its format code, indicating an empty field.'.format(key))
            return np.array([], dtype=format.dtype)
        base = self
        while isinstance(base, FITS_rec) and isinstance(base.base, np.recarray):
            base = base.base
        field = _get_recarray_field(base, name)
        if name not in self._converted:
            recformat = format.recformat
            if isinstance(recformat, _FormatP):
                converted = self._convert_p(column, field, recformat)
            else:
                converted = self._convert_other(column, field, recformat)
            self._cache_field(name, converted)
            return converted
        return self._converted[name]
    def _cache_field(self, name, field):
        base = field
        while True:
            self_base = self
            while True:
                if self_base is base:
                    return
                if getattr(self_base, 'base', None) is not None:
                    self_base = self_base.base
                else:
                    break
            if getattr(base, 'base', None) is not None:
                base = base.base
            else:
                break
        self._converted[name] = field
    def _update_column_attribute_changed(self, column, idx, attr, old_value, new_value):
        method_name = '_update_column_{0}'.format(attr)
        if hasattr(self, method_name):
            getattr(self, method_name)(column, idx, old_value, new_value)
    def _update_column_name(self, column, idx, old_name, name):
        dtype = self.dtype
        dtype.names = dtype.names[:idx] + (name,) + dtype.names[idx + 1:]
    def _convert_x(self, field, recformat):
        dummy = np.zeros(self.shape + (recformat.repeat,), dtype=np.bool_)
        _unwrapx(field, dummy, recformat.repeat)
        return dummy
    def _convert_p(self, column, field, recformat):
        dummy = _VLF([None] * len(self), dtype=recformat.dtype)
        raw_data = self._get_raw_data()
        if raw_data is None:
            raise OSError('Could not find heap data for the {!r} variable-length array column.'.format(column.name))
        for idx in range(len(self)):
            offset = field[idx, 1] + self._heapoffset
            count = field[idx, 0]
            if recformat.dtype == 'a':
                dt = np.dtype(recformat.dtype + str(1))
                arr_len = count * dt.itemsize
                da = raw_data[offset:offset + arr_len].view(dt)
                da = np.char.array(da.view(dtype=dt), itemsize=count)
                dummy[idx] = decode_ascii(da)
            else:
                dt = np.dtype(recformat.dtype)
                arr_len = count * dt.itemsize
                dummy[idx] = raw_data[offset:offset + arr_len].view(dt)
                dummy[idx].dtype = dummy[idx].dtype.newbyteorder('>')
                dummy[idx] = self._convert_other(column, dummy[idx], recformat)
        return dummy
    def _convert_ascii(self, column, field):
        format = column.format
        recformat = ASCII2NUMPY[format[0]]
        nullval = str(column.null).strip().encode('ascii')
        if len(nullval) > format.width:
            nullval = nullval[:format.width]
        dummy = np.char.ljust(field, format.width)
        dummy = np.char.replace(dummy, encode_ascii('D'), encode_ascii('E'))
        null_fill = encode_ascii(str(ASCIITNULL).rjust(format.width))
        dummy = np.where(np.char.strip(dummy) == nullval, null_fill, dummy)
        if nullval != b'':
            dummy = np.where(np.char.strip(dummy) == b'', null_fill, dummy)
        try:
            dummy = np.array(dummy, dtype=recformat)
        except ValueError as exc:
            indx = self.names.index(column.name)
            raise ValueError('{}; the header may be missing the necessary TNULL{} keyword or the table contains invalid data'.format(exc, indx + 1))
        return dummy
    def _convert_other(self, column, field, recformat):
        if isinstance(recformat, _FormatX):
            return self._convert_x(field, recformat)
        (_str, _bool, _number, _scale, _zero, bscale, bzero, dim) = self._get_scale_factors(column)
        indx = self.names.index(column.name)
        if not _str and isinstance(self._coldefs, _AsciiColDefs):
            field = self._convert_ascii(column, field)
        if dim:
            if field.ndim > 1:
                actual_shape = field.shape[1:]
                if _str:
                    actual_shape = actual_shape + (field.itemsize,)
            else:
                actual_shape = field.shape[0]
            if dim == actual_shape:
                dim = None
            else:
                nitems = reduce(operator.mul, dim)
                if _str:
                    actual_nitems = field.itemsize
                elif len(field.shape) == 1:
                    actual_nitems = 1
                else:
                    actual_nitems = field.shape[1]
                if nitems > actual_nitems:
                    warnings.warn('TDIM{} value {:d} does not fit with the size of the array items ({:d}).  TDIM{:d} will be ignored.'.format(indx + 1, self._coldefs[indx].dims, actual_nitems, indx + 1))
                    dim = None
        if not column.ascii and column.format.p_format:
            format_code = column.format.p_format
        else:
            format_code = column.format.format
        if _number and (_scale or _zero) and (not column._physical_values):
            if self._uint:
                if bzero == 2 ** 15 and format_code == 'I':
                    field = np.array(field, dtype=np.uint16)
                elif bzero == 2 ** 31 and format_code == 'J':
                    field = np.array(field, dtype=np.uint32)
                elif bzero == 2 ** 63 and format_code == 'K':
                    field = np.array(field, dtype=np.uint64)
                    bzero64 = np.uint64(2 ** 63)
                else:
                    field = np.array(field, dtype=np.float64)
            else:
                field = np.array(field, dtype=np.float64)
            if _scale:
                np.multiply(field, bscale, field)
            if _zero:
                if self._uint and format_code == 'K':
                    test_overflow = field.copy()
                    try:
                        test_overflow += bzero64
                    except OverflowError:
                        warnings.warn('Overflow detected while applying TZERO{0:d}. Returning unscaled data.'.format(indx + 1))
                    else:
                        field = test_overflow
                else:
                    field += bzero
        elif _bool and field.dtype != bool:
            field = np.equal(field, ord('T'))
        elif _str:
            if not self._character_as_bytes:
                with suppress(UnicodeDecodeError):
                    field = decode_ascii(field)
        if dim:
            nitems = reduce(operator.mul, dim)
            if field.ndim > 1:
                field = field[:, :nitems]
            if _str:
                fmt = field.dtype.char
                dtype = ('|{}{}'.format(fmt, dim[-1]), dim[:-1])
                field.dtype = dtype
            else:
                field.shape = (field.shape[0],) + dim
        return field
    def _get_heap_data(self):
        if self._heapsize:
            raw_data = self._get_raw_data().view(np.ubyte)
            heap_end = self._heapoffset + self._heapsize
            return raw_data[self._heapoffset:heap_end]
        else:
            return np.array([], dtype=np.ubyte)
    def _get_raw_data(self):
        raw_data_bytes = self.nbytes + self._heapsize
        base = self
        while hasattr(base, 'base') and base.base is not None:
            base = base.base
            if hasattr(base, 'nbytes') and base.nbytes >= raw_data_bytes:
                return base
    def _get_scale_factors(self, column):
        _str = column.format.format == 'A'
        _bool = column.format.format == 'L'
        _number = not (_bool or _str)
        bscale = column.bscale
        bzero = column.bzero
        _scale = bscale not in ('', None, 1)
        _zero = bzero not in ('', None, 0)
        if not _scale:
            bscale = 1
        if not _zero:
            bzero = 0
        dim = column._dims
        return (_str, _bool, _number, _scale, _zero, bscale, bzero, dim)
    def _scale_back(self, update_heap_pointers=True):
        heapsize = 0
        for (indx, name) in enumerate(self.dtype.names):
            column = self._coldefs[indx]
            recformat = column.format.recformat
            raw_field = _get_recarray_field(self, indx)
            if isinstance(recformat, _FormatP):
                dtype = np.array([], dtype=recformat.dtype).dtype
                if update_heap_pointers and name in self._converted:
                    raw_field[:] = 0
                    npts = [len(arr) for arr in self._converted[name]]
                    raw_field[:len(npts), 0] = npts
                    raw_field[1:, 1] = np.add.accumulate(raw_field[:-1, 0]) * dtype.itemsize
                    raw_field[:, 1][:] += heapsize
                heapsize += raw_field[:, 0].sum() * dtype.itemsize
            if isinstance(recformat, _FormatX) and name in self._converted:
                _wrapx(self._converted[name], raw_field, recformat.repeat)
                continue
            (_str, _bool, _number, _scale, _zero, bscale, bzero, _) = self._get_scale_factors(column)
            field = self._converted.get(name, raw_field)
            if _number or _str:
                if _number and (_scale or _zero) and column._physical_values:
                    dummy = field.copy()
                    if _zero:
                        dummy -= bzero
                    if _scale:
                        dummy /= bscale
                    column._physical_values = False
                elif _str or isinstance(self._coldefs, _AsciiColDefs):
                    dummy = field
                else:
                    continue
                if isinstance(self._coldefs, _AsciiColDefs):
                    self._scale_back_ascii(indx, dummy, raw_field)
                elif isinstance(raw_field, chararray.chararray):
                    self._scale_back_strings(indx, dummy, raw_field)
                else:
                    if len(raw_field) and isinstance(raw_field[0], np.integer):
                        dummy = np.around(dummy)
                    if raw_field.shape == dummy.shape:
                        raw_field[:] = dummy
                    else:
                        raw_field[:] = dummy.ravel().view(raw_field.dtype)
                del dummy
            elif _bool and name in self._converted:
                choices = (np.array([ord('F')], dtype=np.int8)[0], np.array([ord('T')], dtype=np.int8)[0])
                raw_field[:] = np.choose(field, choices)
        self._heapsize = heapsize
    def _scale_back_strings(self, col_idx, input_field, output_field):
        if input_field.dtype.kind == 'U' and output_field.dtype.kind == 'S':
            try:
                _ascii_encode(input_field, out=output_field)
            except _UnicodeArrayEncodeError as exc:
                raise ValueError("Could not save column '{0}': Contains characters that cannot be encoded as ASCII as required by FITS, starting at the index {1!r} of the column, and the index {2} of the string at that location.".format(self._coldefs[col_idx].name, exc.index[0] if len(exc.index) == 1 else exc.index, exc.start))
        else:
            input_field = input_field.flatten().view(output_field.dtype)
            output_field.flat[:] = input_field
        _rstrip_inplace(output_field)
    def _scale_back_ascii(self, col_idx, input_field, output_field):
        starts = self._coldefs.starts[:]
        spans = self._coldefs.spans
        format = self._coldefs[col_idx].format
        end = super().field(-1).itemsize
        starts.append(end + starts[-1])
        if col_idx > 0:
            lead = starts[col_idx] - starts[col_idx - 1] - spans[col_idx - 1]
        else:
            lead = 0
        if lead < 0:
            warnings.warn('Column {!r} starting point overlaps the previous column.'.format(col_idx + 1))
        trail = starts[col_idx + 1] - starts[col_idx] - spans[col_idx]
        if trail < 0:
            warnings.warn('Column {!r} ending point overlaps the next column.'.format(col_idx + 1))
        if 'A' in format:
            _pc = '{:'
        else:
            _pc = '{:>'
        fmt = ''.join([_pc, format[1:], ASCII2STR[format[0]], '}', ' ' * trail])
        trailing_decimal = format.precision == 0 and format.format in ('F', 'E', 'D')
        for (jdx, value) in enumerate(input_field):
            value = fmt.format(value)
            if len(value) > starts[col_idx + 1] - starts[col_idx]:
                raise ValueError("Value {!r} does not fit into the output's itemsize of {}.".format(value, spans[col_idx]))
            if trailing_decimal and value[0] == ' ':
                value = value[1:] + '.'
            output_field[jdx] = value
        if 'D' in format:
            output_field = output_field.replace(encode_ascii('E'), encode_ascii('D'))
def _get_recarray_field(array, key):
    field = np.recarray.field(array, key)
    if field.dtype.char in ('S', 'U') and (not isinstance(field, chararray.chararray)):
        field = field.view(chararray.chararray)
    return field
class _UnicodeArrayEncodeError(UnicodeEncodeError):
    def __init__(self, encoding, object_, start, end, reason, index):
        super().__init__(encoding, object_, start, end, reason)
        self.index = index
def _ascii_encode(inarray, out=None):
    out_dtype = np.dtype(('S{0}'.format(inarray.dtype.itemsize // 4), inarray.dtype.shape))
    if out is not None:
        out = out.view(out_dtype)
    op_dtypes = [inarray.dtype, out_dtype]
    op_flags = [['readonly'], ['writeonly', 'allocate']]
    it = np.nditer([inarray, out], op_dtypes=op_dtypes, op_flags=op_flags, flags=['zerosize_ok'])
    try:
        for (initem, outitem) in it:
            outitem[...] = initem.item().encode('ascii')
    except UnicodeEncodeError as exc:
        index = np.unravel_index(it.iterindex, inarray.shape)
        raise _UnicodeArrayEncodeError(*exc.args + (index,))
    return it.operands[1]
def _has_unicode_fields(array):
    dtypes = (d[0] for d in array.dtype.fields.values())
    return any((d.kind == 'U' for d in dtypes))